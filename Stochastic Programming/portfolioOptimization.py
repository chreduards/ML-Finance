import quantopian.algorithm as algo # Import Algorithm API
import pandas as pd
import math
import numpy as np
from quantopian.pipeline import Pipeline

def initialize(context):
    
    context.interestRate = 0.02 # Risk free interest rate
    context.marketExcessPremium = 0.02 # Market excess premium used in CAPM
    context.borrowing = 'true'
    context.shorting = 'true'
    context.objGamma = 0.5 # Objective function gamma
    context.t = 1 # Time step used in monte carlo simulation
    context.nSamples = 10 # Number of scenarios in monte-carlo

def before_trading_start(context,data):
    
    # Defines the trading universe
    context.stocks = symbols('ALL', 'AEE', 'ADP', 'BA', 'COF') 
    
    # Time between the historical returns
    dt = 1 / 252 
    
    # Fetches historical prices
    context.historicalPrices = data.history(context.stocks, fields='price', bar_count=3000, frequency='1d') 
    
    logReturns = pd.DataFrame(computeLogReturns(context,data)) 
    # Computes historical average returns
    nuHist = pd.DataFrame(estExpected(logReturns,dt)) 
    sigma, corr, cov = estVolEWMA(logReturns,dt) 
    
    marketCapWeight = np.array(np.ones(len(context.stocks))/len(context.stocks))[np.newaxis].T
    # Computes expected returns
    nuCAPM = estExpectedCAPM(cov, context.marketExcessPremium, context.interestRate, marketCapWeight) 
    
    nu = estExpectedBlackLitterman(nuCAPM, cov*0.1, np.asmatrix(np.identity(len(context.stocks))), nuHist, cov)
    
    # Generates scenarioPrices with Monte Carlo simulation
    scenarioPrices = genScenariosRegular(nu, sigma, corr, context.t, len(context.stocks), context.nSamples) 
       
def genScenariosRegular(nu, sigma, corr, t, nAssets, nSamples):

    # Compute the cholesky factorisation to preserve the correlation in the data
    C = np.matrix(np.linalg.cholesky(corr)) 
    # Compute normally distributed values with the same correlation as nu
    xi = (C * np.matrix(np.random.normal(0, 1, (nAssets, nSamples)))).T    
    # Generate prices
    prices = np.exp(np.tile(nu * t, (nSamples, 1)) + np.multiply(np.tile(sigma * np.sqrt(t), (nSamples, 1)), xi))        
    return prices

# Uses the Black-Litterman method to compute expected returns
# Called by: before_trading_starts
def estExpectedBlackLitterman(muP, Sigma, P, q, Omega):
    H = np.linalg.inv(Sigma) + P.T * np.linalg.solve(Omega, P)
    mu = np.linalg.solve(H, np.linalg.solve(Sigma, muP) + P.T * (np.linalg.solve(Omega, q)))
    return np.array(mu).T
       
# Computes the expected returns using CAPM
# Called by: before_trading_starts
def estExpectedCAPM(cov, marketExcessPremium, interestRate, marketCapWeight):    
    
    beta = cov * marketCapWeight / (marketCapWeight.T * cov * marketCapWeight)
    mu = np.exp(interestRate) - 1 + beta * marketExcessPremium
    sigma = np.array(np.sqrt(np.diag(cov)))[np.newaxis].T
    nu = np.array((mu - np.square(sigma) / 2)) # Drift
    return nu

# Computes the yearly expected returns
# Called by: before_trading_start
def estExpected(logReturns,dt):
    estPeriod = 5 # Timeframe in years for historical returns
    activeReturns = determineActiveReturns(logReturns, estPeriod, dt)
    estExpected = activeReturns.mean(axis = 0, skipna = True) / dt
    return estExpected

# Computes log returns based on historical prices for stocks defined in context.stocks
# Called by: before_trading_start()
def computeLogReturns(context,data):
    logReturns = pd.DataFrame()
    tempLogReturns = pd.DataFrame()
    for i in range(len(context.historicalPrices.columns)):
            tempLogReturns = np.log10(1 + 
            context.historicalPrices.iloc[:,i].pct_change())
            logReturns.insert(loc = i, column = context.stocks[i], value = tempLogReturns[1:len(tempLogReturns)])
    return logReturns

# Computes the active returns based on the estPeriod timeframe variable
# Called by: estExpected(), estVolEWMA()
def determineActiveReturns(logReturns, estPeriod, dt):
    firstDate = len(logReturns) - math.ceil(estPeriod/dt)
    lastDate = len(logReturns)
    return logReturns.iloc[firstDate:lastDate,:]

# Computes the correlation matrix and standard deviation of the logReturns
# Called by: before_trading_start
def estVolEWMA(logReturns,dt):
    estPeriod = 1 # Timeframe in years for volatility estimation
    lbda = 0.95 # EWMA lambda value
    activeReturns = pd.DataFrame(determineActiveReturns(logReturns, estPeriod, dt))
    
    cov = activeReturns.cov() # Initialize the covariance matrix
    for i in range(len(activeReturns)):
        loopActiveReturns = np.array(activeReturns.iloc[i,:])[np.newaxis]
        cov = lbda * cov + (1 - lbda) * loopActiveReturns.T * loopActiveReturns / dt # Compute EWMA
    
    sigma = np.array(np.sqrt(np.diag(cov)))[np.newaxis] 
    corr = np.divide(cov,sigma.T*sigma)
    return sigma, np.matrix(corr), np.matrix(cov)


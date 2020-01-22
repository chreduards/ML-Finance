# Import Algorithm API
import quantopian.algorithm as algo
import pandas as pd
import math
import numpy
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing 
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.filters.morningstar import Q500US

def initialize(context):
 
    #historical_prices = schedule_function(get_historical_prices,date_rules.every_day())
    weights = {}
    

def before_trading_start(context,data):

    stocks = symbols('ALL', 'AEE', 'ADP', 'BA', 'COF')
    
    context.historical_prices = data.history(stocks, fields='price', bar_count=3000, frequency='1d')   
    
    log_returns = pd.DataFrame(computeLogReturns(context,data))
    nu_hist = estExpected(log_returns)
    print(nu_hist)
    
#Computes the yearly expected returns
#Called by: before_trading_start
def estExpected(log_returns):
    dt = 1 / 252
    estExpected = log_returns.mean(axis = 1, skipna = True) / dt
    return estExpected

# Computes log returns based on historical prices for stocks defined in context.stocks
# Called by: before_trading_start()
def computeLogReturns(context,data):
    log_returns = pd.DataFrame()
    temp_log_returns = pd.DataFrame()
    for i in range(len(context.historical_prices.columns)):
            temp_log_returns = numpy.log10(1 + context.historical_prices.iloc[:,i].pct_change())
            log_returns = log_returns.append(temp_log_returns)
    return log_returns

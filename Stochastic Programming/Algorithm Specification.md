# Portfolio optimization algoritm
The algoritm will be constructed in steps. The goal is to rebalance an equity portfolio on a daily basis.
This algoritm will act as a proof of concept and will be the first step in a larger implementation.

Overview of Steps:
#### Implement Black-Litterman using 
- [x] Historical returns 
- [x] Expected returns from CAPM
- [x] EWMA (Need volatility of historical returns to calculate the covariance of the historical returns)
- [x] Implement Black-Litterman

#### Generate scenarios with Monte Carlo 
- [ ] Generate values without variance reduction
- [ ] Generate values with Latin Hypercube
- [ ] Generate scenario prices

#### Implement a primal dual solver
- [ ] Implement solver

#### Rebalance the portfolio
- [ ] Rebalance portfolio

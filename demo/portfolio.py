# get data
import dafin

asset_lsit = ["SPY", "GLD", "BND"]
date_start = "2010-01-01"
date_end = "2022-12-30"

data_instance = dafin.ReturnsData(asset_lsit)
returns_data = data_instance.get_returns(date_start, date_end)

# portfolio
from portfawn import (
    BackTest,
    RandomPortfolio,
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
)

portfolio_list = [
    RandomPortfolio(),
    EquallyWeightedPortfolio(),
    MeanVariancePortfolio(),
]

backtest = BackTest(
    portfolio_list=portfolio_list,
    asset_list=asset_lsit,
    date_start=date_start,
    date_end=date_end,
    fitting_days=252,
    evaluation_days=22,
    n_jobs=12,
)
backtest.run()


import matplotlib.pyplot as plt

fig, ax = backtest.plot_returns()
plt.savefig("random_portfolio_returns.png")

fig, ax = backtest.plot_cum_returns()
plt.savefig("random_portfolio_cum_returns.png")


fig, ax = backtest.plot_dist_returns()
plt.savefig("random_portfolio_dist_returns.png")


fix, ax = backtest.plot_corr()
plt.savefig("random_portfolio_corr.png")

# plot_cov
fig, ax = backtest.plot_cov()
plt.savefig("random_portfolio_cov.png")


# # equally weighted
# equally_weighted_portfolio = portfawn.portfolio.EquallyWeightedPortfolio()
# equally_weighted_portfolio.fit(returns_data)
# performance = equally_weighted_portfolio.evaluate(returns_data)
# print(performance)

# # mean variance
# mean_var_portfolio = portfawn.portfolio.MeanVariancePortfolio()
# mean_var_portfolio.fit(returns_data)
# performance = mean_var_portfolio.evaluate(returns_data)
# print(performance)

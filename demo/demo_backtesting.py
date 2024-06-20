import dafin
import matplotlib.pyplot as plt

from portfawn import (
    BackTest,
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    RandomPortfolio,
)

# params
# asset_lsit = ["SPY", "GLD", "BND"]
asset_lsit = [
    "AAPL",
    "ORCL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "AVGO",
    "JPM",
    "V",
    "WMT",
    "XOM",
    "UNH",
]

date_start = "2010-01-01"
date_end = "2022-12-30"
data_instance = dafin.ReturnsData(asset_lsit)
returns_data = data_instance.get_returns(date_start, date_end)

portfolio_list = [
    RandomPortfolio(),
    EquallyWeightedPortfolio(),
    MeanVariancePortfolio(),
]

# backtesting
backtest = BackTest(
    portfolio_list=portfolio_list,
    asset_list=asset_lsit,
    date_start=date_start,
    date_end=date_end,
    fitting_days=252,
    evaluation_days=90,
    n_jobs=12,
)
backtest.run()

# visualization
fig, ax = backtest.plot_returns()
plt.savefig("plot_returns.png")

fig, ax = backtest.plot_cum_returns()
plt.savefig("plot_cum_returns.png")

fig, ax = backtest.plot_dist_returns()
plt.savefig("plot_dist_returns.png")

fix, ax = backtest.plot_corr()
plt.savefig("plot_corr.png")

fig, ax = backtest.plot_cov()
plt.savefig("plot_cov.png")

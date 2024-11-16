import logging

import dafin
import matplotlib.pyplot as plt

from portfawn import (
    BackTest,
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    OptimizationModel,
    RandomPortfolio,
)

logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] (%(name)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    level=logging.WARNING,
)

logger = logging.getLogger(__name__)


# params
asset_lsit = ["SPY", "GLD", "BND"]
date_start = "2010-01-01"
date_end = "2022-12-30"
data_instance = dafin.ReturnsData(asset_lsit)
returns_data = data_instance.get_returns(date_start, date_end)

mean_vafiance_portfolio = [
    MeanVariancePortfolio(name=o, optimization_model=OptimizationModel(objective=o))
    for o in ["MVP", "MSRP", "BMOP"]
]
portfolio_list = [
    RandomPortfolio(),
    EquallyWeightedPortfolio(),
]
portfolio_list.extend(mean_vafiance_portfolio)

# backtesting
backtest = BackTest(
    portfolio_list=portfolio_list,
    asset_list=asset_lsit,
    date_start=date_start,
    date_end=date_end,
    fitting_days=22,
    evaluation_days=5,
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

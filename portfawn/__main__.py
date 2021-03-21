from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from portfawn.portfolio import Portfolio, PortfolioBackTesting, PlotPortfolio
from portfawn.plot import Plot
from portfawn.utils import remove_one_asset_portfolio
from portfawn.market_data import MarketData, PlotMarketData

logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] (%(name)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    level=logging.ERROR,
)

logger = logging.getLogger(__name__)


def main():

    # parameters
    us_bonds = ["BLV", "BIV", "BSV", "VCLT", "VCIT", "VCSH"]
    us_stocks = ["MGC", "VV", "VO", "VB"]
    int_stocks = ["VEU", "VWO"]
    asset_list = us_bonds + us_stocks + int_stocks
    start = datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    end = datetime.strptime("2020-12-31", "%Y-%m-%d").date()
    training_days = 22
    testing_days = 5
    risk_free_rate = 0.0
    portfolio_names = [
        "equal",
        "min_variance",
        "max_return",
        "max_sharpe_ratio",
        "binary_qpu",
        "binary_sa",
    ]

    # download the data
    market_data = MarketData(
        asset_list=asset_list,
        date_start=start - pd.Timedelta(training_days, unit="d"),
        date_end=end + pd.Timedelta(testing_days, unit="d"),
    )
    PlotMarketData(market_data)

    # backtesting
    portfolio_backtesting = PortfolioBackTesting(
        asset_list=asset_list,
        portfolio_names=portfolio_names,
        start_date_analysis=start,
        end_date_analysis=end,
        training_days=training_days,
        testing_days=testing_days,
        risk_free_rate=risk_free_rate,
        path_data=Path("data"),
        path_results=Path("results"),
    )
    portfolio_backtesting.analyze()
    portfolio_backtesting.plot()


def demo_data_market():
    us_bonds = ["BLV", "BIV", "BSV", "VCLT", "VCIT", "VCSH"]
    us_stocks = ["MGC", "VV", "VO", "VB"]
    int_stocks = ["VEU", "VWO"]
    asset_list = us_bonds + us_stocks + int_stocks
    date_start = datetime.strptime("2010-01-01", "%Y-%m-%d").date()
    date_end = datetime.strptime("2019-12-31", "%Y-%m-%d").date()

    market_data = MarketData(
        asset_list=asset_list, date_start=date_start, date_end=date_end
    )

    PlotMarketData(market_data)


if __name__ == "__main__":
    main()
    # demo_data_market()

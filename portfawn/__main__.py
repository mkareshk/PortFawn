from pathlib import Path
from datetime import datetime
import logging

from portfawn.portfolio import PortfolioBackTesting

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
    asset_list = ["BND", "GLD", "SPY"]
    start = datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    end = datetime.strptime("2020-01-30", "%Y-%m-%d").date()
    training_days = 22
    testing_days = 5
    risk_free_rate = 0.0
    portfolio_types = [
        "equal",
        "random"
        # "min_variance",
        # "max_return",
        # "max_sharpe_ratio",
        # "binary_qpu",
        # "binary_sa",
    ]
    sampling_methods = ["simple"]
    optimization_methods = ["equal", "random"]

    # backtesting
    portfolio_backtesting = PortfolioBackTesting(
        asset_list=asset_list,
        portfolio_types=portfolio_types,
        sampling_methods=sampling_methods,
        optimization_methods=optimization_methods,
        start_date_analysis=start,
        end_date_analysis=end,
        training_days=training_days,
        testing_days=testing_days,
        risk_free_rate=risk_free_rate,
        path_data=Path("data"),
        path_results=Path("results"),
    )
    portfolio_backtesting.run()


if __name__ == "__main__":
    main()

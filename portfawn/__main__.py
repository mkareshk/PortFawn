import logging
from datetime import datetime

import joblib

from portfawn.portfolio import BackTesting, BackTestAnalysis


logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] (%(name)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main():

    # parameters
    us_bonds = ["BLV", "BIV", "BSV", "VCLT", "VCIT", "VCSH"]
    us_stocks = ["MGC", "VV", "VO", "VB"]
    int_stocks = ["VEU", "VWO"]
    asset_list = us_bonds + us_stocks + int_stocks

    start_date = datetime.strptime("2010-01-01", "%Y-%m-%d").date()
    end_date = datetime.strptime("2020-01-30", "%Y-%m-%d").date()

    training_days = 22
    testing_days = 5

    risk_free_rate = 0.0

    portfolio_types = [
        "Equal",
        "Random",
        "MV",
        "MR",
        "MSR",
        # "binary_qpu",
        # "binary_sa",
    ]

    core_num = joblib.cpu_count()

    kwargs = {
        "portfolio_types": portfolio_types,
        "asset_list": asset_list,
        "start_date": start_date,
        "end_date": end_date,
        "optimization_params": {"name": "simple"},  ## TODO: remove
        "sampling_params": {"name": "simple"},
        "training_days": training_days,
        "testing_days": testing_days,
        "risk_free_rate": risk_free_rate,
        "n_jobs": core_num - 1,
    }

    # backtesting
    portfolio_backtesting = BackTesting(**kwargs)
    portfolio_backtesting.run()

    # analysis
    analysis = BackTestAnalysis(portfolio_backtesting)
    analysis.plot()


if __name__ == "__main__":
    main()

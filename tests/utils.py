from datetime import datetime

import joblib


def get_normal_param():

    # tickers
    us_bonds = ["BLV", "BIV", "BSV", "VCLT", "VCIT", "VCSH"]
    us_stocks = ["MGC", "VV", "VO", "VB"]
    int_stocks = ["VEU", "VWO"]
    asset_list = us_bonds + us_stocks + int_stocks

    # date
    start_date = datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    end_date = datetime.strptime("2020-01-30", "%Y-%m-%d").date()

    training_days = 22
    testing_days = 5

    # market
    risk_free_rate = 0.0

    # portfolio
    portfolio_types = [
        "Equal",
        "Random",
        "MV",
        "MR",
        "MSR",
        # "binary_qpu",
        # "binary_sa",
    ]

    # system
    core_num = joblib.cpu_count()

    return {
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

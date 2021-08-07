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
    end_date = datetime.strptime("2021-06-30", "%Y-%m-%d").date()

    training_days = 60
    testing_days = 30

    # market
    risk_free_rate = 0.0

    # portfolio
    portfolio_types = ["SA", "MV", "MR", "MSR", "Equal"]
    optimization_params = {
        "scipy_params": {
            "maxiter": 1000,
            "disp": False,
            "ftol": 1e-10,
        },
        "target_return": 0.1,
        "target_risk": 0.1,
        "weight_bound": (0.0, 1.0),
    }
    sampling_params = {"type": "standard"}

    # system
    n_jobs = joblib.cpu_count() - 1

    return {
        "portfolio_types": portfolio_types,
        "asset_list": asset_list,
        "start_date": start_date,
        "end_date": end_date,
        "optimization_params": optimization_params,
        "sampling_params": sampling_params,
        "training_days": training_days,
        "testing_days": testing_days,
        "risk_free_rate": risk_free_rate,
        "n_jobs": n_jobs,
    }

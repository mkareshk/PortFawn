import logging
import copy

import numpy as np
import pandas as pd

from portfawn.portfolio.portfolio import MeanVariancePortfolio
from portfawn.portfolio.utils import random_portfolio


logger = logging.getLogger(__name__)


class MultiPortfolio:
    def __init__(
        self,
        name: str,
        objectives_list: list,
        risk_type="standard",
        risk_sample_num=100,
        risk_sample_size=20,
        risk_agg_func="median",
        risk_free_rate=0,
        annualized_days=252,
        qpu_params={"backend": "neal", "annealing_time": 100},
        scipy_params={"maxiter": 1000, "disp": False, "ftol": 1e-10},
        target_return=0.1,
        target_sd=0.1,
        weight_bound=(0.0, 1.0),
        init_point=None,
    ):

        # args
        self.name = name
        self.objectives_list = objectives_list
        self.risk_type = risk_type
        self.risk_sample_num = risk_sample_num
        self.risk_sample_size = risk_sample_size
        self.risk_agg_func = risk_agg_func
        self.risk_free_rate = risk_free_rate
        self.annualized_days = annualized_days
        self.qpu_params = qpu_params
        self.scipy_params = scipy_params
        self.target_return = target_return
        self.target_sd = target_sd
        self.weight_bound = weight_bound
        self.init_point = init_point

        self.portfolio_config = {
            "name": name,
            "risk_type": risk_type,
            "risk_sample_num": risk_sample_num,
            "risk_sample_size": risk_sample_size,
            "risk_agg_func": risk_agg_func,
            "risk_free_rate": risk_free_rate,
            "annualized_days": annualized_days,
            "target_return": target_return,
            "target_sd": target_sd,
            "weight_bound": weight_bound,
            "init_point": init_point,
        }
        self.portfolios = {}

        for objective in self.objectives_list:

            param = copy.deepcopy(self.portfolio_config)
            param.update({"objective": objective})

            self.portfolios[objective] = MeanVariancePortfolio(**param)

    def run(self, asset_list, date_start="2010-01-01", date_end="2021-12-31"):

        mean_sd_list = []

        for o in self.objectives_list:
            self.portfolios[o].fit(asset_list, date_start, date_end)

        portfolio_results_list = [
            self.portfolios[o].evaluate(date_start, date_end)
            for o in self.objectives_list
        ]

        mean_sd_list = [r["annualized_mean_sd"] for r in portfolio_results_list]

        mean_sd = pd.concat(mean_sd_list, axis=0)
        portfolio_mean_sd = mean_sd.loc[self.objectives_list, :]
        market_mean_sd = mean_sd.loc[asset_list, :]

        # random portfolios
        mean_sd_random = random_portfolio(
            returns=portfolio_results_list[0]["returns"].loc[:, asset_list],
            days_per_year=portfolio_results_list[0]["days_per_year"],
            annualized=True,
        )

        return {
            "market_mean_sd": market_mean_sd,
            "portfolio_mean_sd": portfolio_mean_sd,
            "mean_sd_random": mean_sd_random,
        }

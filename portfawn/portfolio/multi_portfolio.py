import logging
import copy

import numpy as np
import pandas as pd

from portfawn.portfolio.portfolio import MeanVariancePortfolio  # , PortfolioParams

logger = logging.getLogger(__name__)


# class MultiPortfolioParams(PortfolioParams):
#     def __init__(
#         self,
#         name: str,
#         objectives_list: list,
#         risk_type="standard",
#         risk_sample_num=100,
#         risk_sample_size=20,
#         risk_agg_func="median",
#         risk_free_rate=0,
#         annualized_days=252,
#         backend="neal",
#         annealing_time=100,
#         scipy_params={"maxiter": 1000, "disp": False, "ftol": 1e-10},
#         target_return=0.1,
#         target_sd=0.1,
#         weight_bound=(0.0, 1.0),
#         init_point=None,
#     ):

#         self.objectives_list = objectives_list

#         super().__init__(
#             name,
#             risk_type=risk_type,
#             risk_sample_num=risk_sample_num,
#             risk_sample_size=risk_sample_size,
#             risk_agg_func=risk_agg_func,
#             risk_free_rate=risk_free_rate,
#             annualized_days=annualized_days,
#             backend=backend,
#             annealing_time=annealing_time,
#             scipy_params=scipy_params,
#             target_return=target_return,
#             target_sd=target_sd,
#             weight_bound=weight_bound,
#             init_point=init_point,
#         )


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
            # print(param)

            self.portfolios[objective] = MeanVariancePortfolio(**param)

    def run(self, asset_list, date_start="2010-01-01", date_end="2021-12-31"):

        mean_sd_list = []

        for o in self.objectives_list:
            self.portfolios[o].fit(asset_list, date_start, date_end)

        portfolio_results_list = [
            self.portfolios[o].evaluate(date_start, date_end)
            for o in self.objectives_list
        ]
        annualized_days = portfolio_results_list[0]["portfolio_config"][
            "annualized_days"
        ]
        mean_sd_list = [r["mean_sd"] for r in portfolio_results_list]

        mean_sd = pd.concat(mean_sd_list, axis=0)
        mean_sd["mean"] *= annualized_days
        mean_sd["sd"] *= np.sqrt(annualized_days)
        portfolio_mean_sd = mean_sd.loc[self.objectives_list, :]
        market_mean_sd = mean_sd.loc[asset_list, :]

        # random portfolios
        n = 1000
        returns_df = portfolio_results_list[0]["returns"].loc[:, asset_list]
        returns_np = returns_df.to_numpy()
        cov = returns_df.cov().to_numpy()

        r_list = []
        for i in range(n):
            w_rand = np.random.random((1, cov.shape[0]))
            w_rand = w_rand / w_rand.sum()
            r = returns_np.dot(w_rand.T).mean() * annualized_days
            c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0] * np.sqrt(annualized_days)
            r_list.append({"mean": r, "sd": c})
        mean_sd_random = pd.DataFrame(r_list)

        return {
            "market_mean_sd": market_mean_sd,
            "portfolio_mean_sd": portfolio_mean_sd,
            "mean_sd_random": mean_sd_random,
        }

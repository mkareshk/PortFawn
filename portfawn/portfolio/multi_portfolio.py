import logging

import numpy as np
import pandas as pd

from portfawn.portfolio.portfolio import Portfolio

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
        risk_free_rate=0.0,
        annualized_days=252,
        backend="neal",
        annealing_time=100,
        scipy_params={"maxiter": 1000, "disp": False, "ftol": 1e-10},
        target_return=0.1,
        target_sd=0.1,
        weight_bound=(0.0, 1.0),
        init_point=None,
    ):

        # args
        self._name = name
        self._objectives_list = objectives_list
        self._risk_type = risk_type
        self._risk_sample_num = risk_sample_num
        self._risk_sample_size = risk_sample_size
        self._risk_agg_func = risk_agg_func
        self._risk_free_rate = risk_free_rate
        self._annualized_days = annualized_days
        self._backend = backend
        self._annealing_time = annealing_time
        self._scipy_params = scipy_params
        self._target_return = target_return
        self._target_sd = target_sd
        self._weight_bound = weight_bound
        self._init_point = init_point

        self.portfolios = {}

        for objective in self._objectives_list:
            self.portfolios[objective] = Portfolio(
                name=objective,
                objective=objective,
                risk_type=self._risk_type,
                risk_sample_num=self._risk_sample_num,
                risk_sample_size=self._risk_sample_size,
                risk_agg_func=self._risk_agg_func,
                risk_free_rate=self._risk_free_rate,
                annualized_days=self._annualized_days,
                backend=self._backend,
                annealing_time=self._annealing_time,
                scipy_params=self._scipy_params,
                target_return=self._target_return,
                target_sd=self._target_sd,
                weight_bound=self._weight_bound,
                init_point=self._init_point,
            )

    def run(self, asset_list, date_start="2010-01-01", date_end="2021-12-31"):

        mean_sd_list = []
        portfolio_results_list = [
            self.portfolios[o].run(asset_list, date_start, date_end)
            for o in self._objectives_list
        ]
        annualized_days = portfolio_results_list[0]["portfolio_config"][
            "annualized_days"
        ]
        mean_sd_list = [r["portfolio_mean_sd"] for r in portfolio_results_list]

        portfolio_mean_sd = pd.concat(mean_sd_list, axis=0)
        portfolio_mean_sd["mean"] *= annualized_days
        portfolio_mean_sd["sd"] *= np.sqrt(annualized_days)

        market_mean_sd = portfolio_results_list[0]["market_mean_sd"]
        market_mean_sd["mean"] *= annualized_days
        market_mean_sd["sd"] *= np.sqrt(annualized_days)

        # random portfolios
        n = 1000
        returns_np = portfolio_results_list[0]["asset_returns"].to_numpy()
        cov = portfolio_results_list[0]["asset_returns"].cov().to_numpy()

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

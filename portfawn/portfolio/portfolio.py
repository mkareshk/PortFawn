import logging

import numpy as np
import pandas as pd
from dafin import Returns

from portfawn.models.risk import RiskModel
from portfawn.models.optimization import QuantumOptModel, ClassicOptModel
from portfawn.models.economic import EconomicModel

logger = logging.getLogger(__name__)


class MeanVariancePortfolio:
    def __init__(
        self,
        name: str = "",
        objective: str = "",
        risk_type="standard",
        risk_sample_num=100,
        risk_sample_size=20,
        risk_agg_func="median",
        risk_free_rate=0.0,
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
        self.objective = objective
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
            "objective": objective,
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
        self.portfolio_config.update(qpu_params)
        self.portfolio_config.update(scipy_params)

        # risk model
        self._risk_model = RiskModel(
            type=self.risk_type,
            sample_num=self.risk_sample_num,
            sample_size=self.risk_sample_size,
            agg_func=self.risk_agg_func,
        )

        # economic model
        self._economic_model = EconomicModel(
            risk_free_rate=self.risk_free_rate,
            annualized_days=self.annualized_days,
        )

        # optimization model
        if self.objective in ["BMOP"]:
            self.optimizer = QuantumOptModel(
                objective=self.objective,
                backend=qpu_params["backend"],
                annealing_time=qpu_params["annealing_time"],
            )

        elif self.objective in ["EWP", "MRP", "MVP", "MSRP"]:
            self.optimizer = ClassicOptModel(
                objective=self.objective,
                economic_model=self._economic_model,
                scipy_params=self.scipy_params,
                target_return=self.target_return,
                target_sd=self.target_sd,
                weight_bound=self.weight_bound,
                init_point=self.init_point,
            )

        else:
            raise NotImplementedError

    def fit(self, asset_list, date_start="2010-01-01", date_end="2021-12-31"):

        # returns data
        self.asset_list = asset_list
        self.returns_data = Returns(
            asset_list=asset_list,
            date_start=date_start,
            date_end=date_end,
        )

        self.asset_returns = self.returns_data.returns

        # risk evaluation
        expected_return, expected_cov = self._risk_model.evaluate(self.returns_data)

        # optimization
        self._w = self.optimizer.optimize(
            expected_return=expected_return, expected_cov=expected_cov
        )
        self.asset_weights = {
            asset_list[ind]: float(w) for ind, w in enumerate(self._w)
        }

    def evaluate(self, date_start="2010-01-01", date_end="2021-12-31"):

        if not (self.asset_list and self.asset_weights):
            raise ValueError

        # returns data
        returns_data = Returns(
            asset_list=self.asset_list,
            date_start=date_start,
            date_end=date_end,
        )
        asset_returns = self.returns_data.returns
        asset_cum_returns = returns_data.cum_returns

        # risk evaluation
        expected_return, expected_cov = self._risk_model.evaluate(self.returns_data)
        expected_return_np = expected_return.to_numpy()
        expected_cov_np = expected_cov.to_numpy()

        ## performance metrics

        # portfolio daily returns
        portfolio_returns = pd.DataFrame(
            asset_returns.to_numpy().dot(self._w),
            index=asset_returns.index,
            columns=[self.objective],
        )
        portfolio_assets_returns = pd.concat([asset_returns, portfolio_returns], axis=1)

        # portfolio cummulative return
        portfolio_cum_returns = (portfolio_returns + 1).cumprod() - 1
        portfolio_assets_cum_returns = pd.concat(
            [asset_cum_returns, portfolio_cum_returns], axis=1
        )

        # total returns

        portfolio_asset_total_return = portfolio_assets_cum_returns.iloc[-1, :]
        portfolio_total_return = portfolio_asset_total_return[self.objective]

        # portfolio expected return and sd
        portfolio_expected_return = expected_return_np.dot(self._w)
        portfolio_expected_sd = np.sqrt(self._w.T.dot(expected_cov_np).dot(self._w))

        # market
        market_mean_sd = pd.DataFrame(columns=["mean", "sd"])
        market_mean_sd["mean"] = returns_data.returns.mean()
        market_mean_sd["sd"] = returns_data.returns.std()

        # portfolio
        portfolio_mean_sd = pd.DataFrame(index=[self.objective], columns=["mean", "sd"])
        portfolio_mean_sd["mean"] = portfolio_expected_return
        portfolio_mean_sd["sd"] = portfolio_expected_sd

        performance = {}
        performance.update(
            {
                "asset_weights": self.asset_weights,
                "asset_returns": asset_returns,
                "portfolio_returns": portfolio_returns,
                "portfolio_assets_returns": portfolio_assets_returns,
                "portfolio_cum_returns": portfolio_cum_returns,
                "portfolio_assets_cum_returns": portfolio_assets_cum_returns,
                "portfolio_asset_total_return": portfolio_asset_total_return,
                "portfolio_total_return": portfolio_total_return,
                "portfolio_expected_return": portfolio_expected_return,
                "portfolio_expected_sd": portfolio_expected_sd,
                "market_mean_sd": market_mean_sd,
                "portfolio_mean_sd": portfolio_mean_sd,
                "portfolio_config": self.portfolio_config,
            }
        )

        return performance

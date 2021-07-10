import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from portfawn.sampling import Sampling
from portfawn.market_data import MarketData, MarketDataAnalysis
from portfawn.plot import Plot
from portfawn.portfolio_optimization import PortfolioOptimization
from portfawn.utils import get_assets_signature, is_jsonable

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(
        self,
        portfolio_type,
        data_returns,
        asset_weights=None,
        risk_free_rate=0.0,
        optimization_params=None,
        sampling_params=None,
    ):
        # args
        self.portfolio_type = portfolio_type
        self.optimization_params = optimization_params
        self.sampling_params = sampling_params
        self.data_returns = data_returns
        self.asset_weights = asset_weights
        self.risk_free_rate = risk_free_rate

        # other params
        self.asset_list = list(data_returns.columns)

        self.date_start = data_returns.index[0]
        self.date_end = data_returns.index[-1]

    def optimize(self):
        if self.asset_weights:
            raise Exception(
                f"The portfolio weights have already set, {self.asset_weights}"
            )

        # sampling
        self.expected_stats = Sampling(
            data_returns=self.data_returns, sampling_params=self.sampling_params
        )
        expected_return = self.expected_stats.expected_return
        expected_risk = self.expected_stats.expected_risk

        # optimization
        self.optimizer = PortfolioOptimization(
            self.portfolio_type,
            expected_return=expected_return,
            expected_risk=expected_risk,
            risk_free_rate=self.risk_free_rate,
            optimization_params=self.optimization_params,
        )
        self.asset_weights = self.optimizer.optimize()

    def evaluate(self):

        self.asset_weights_dict = {
            self.asset_list[ind]: float(w) for ind, w in enumerate(self.asset_weights)
        }
        w = self.asset_weights

        performance = {}

        performance.update(
            {f"daily_return": self.data_returns.mean().dot(w).tolist()[0]}
        )

        self.performance = performance

        # "portfolio_types": portfolio_types,
        # "asset_list": asset_list,
        # "start_date": start_date,
        # "end_date": end_date,
        # "optimization_params": {"name": "simple"},  ## TODO: remove
        # "sampling_params": {"name": "simple"},
        # "training_days": training_days,
        # "testing_days": testing_days,
        # "risk_free_rate": risk_free_rate,


class BackTesting:
    def __init__(
        self,
        portfolio_types,
        asset_list,
        start_date,
        end_date,
        optimization_params,
        sampling_params,
        training_days,
        testing_days,
        risk_free_rate,
        n_jobs,
    ):
        """[summary]

        Args:
            portfolio_types ([type]): [description]
            asset_list ([type]): [description]
            start_date ([type]): [description]
            end_date ([type]): [description]
            optimization_params ([type]): [description]
            sampling_params ([type]): [description]
            training_days ([type]): [description]
            testing_days ([type]): [description]
            risk_free_rate (float): [description]
            n_jobs ([type]): [description]
        """

        # parameters
        self.portfolio_types = portfolio_types
        self.asset_list = asset_list
        self.start_date = start_date
        self.end_date = end_date
        self.optimization_params = optimization_params
        self.sampling_params = sampling_params
        self.training_days = training_days
        self.testing_days = testing_days
        self.risk_free_rate = risk_free_rate
        self.n_jobs = n_jobs

        # creating the time windows
        self.analysis_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f"{self.testing_days}D",
        )

        # each window: (the first day of training, the reference day, the last day of testing)
        self.training_delta = pd.Timedelta(self.training_days, unit="d")
        self.testing_delta = pd.Timedelta(self.testing_days, unit="d")
        self.analysis_windows = [
            (i.date() - self.training_delta, i.date(), i.date() + self.testing_delta)
            for i in self.analysis_range
        ]

        # market data
        self.market_data_source = MarketData(
            asset_list=self.asset_list,
            date_start=self.start_date - pd.Timedelta(training_days, unit="d"),
            date_end=self.end_date + pd.Timedelta(training_days, unit="d"),
        )
        # self.market_data_source.collect()
        self.data_returns = self.market_data_source.data_returns

        market_returns_plot = MarketDataAnalysis(
            self.market_data_source, Path("results") / Path("market")
        )
        market_returns_plot.plot()

    def run(self):

        # sequential
        if self.n_jobs == 1:
            profile_backtesting = [
                self.run_iter(**instance) for instance in self.get_portfolio_instances()
            ]

        # parallel
        elif self.n_jobs > 1:
            profile_backtesting = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_iter)(**instance)
                for instance in self.get_portfolio_instances()
            )

        self.profile_backtesting = profile_backtesting

    def get_portfolio_instances(self):
        return [
            dict(
                portfolio_type=portfolio_type,
                date_start_training=window[0],
                date_end_training=window[1],
                date_start_testing=window[1],
                date_end_testing=window[2],
            )
            for window in self.analysis_windows
            for portfolio_type in self.portfolio_types
        ]

    def run_iter(
        self,
        portfolio_type,
        date_start_training,
        date_end_training,
        date_start_testing,
        date_end_testing,
    ):

        # training
        t0 = time.time()

        portfolio_training = self.train(
            portfolio_type=portfolio_type,
            date_start_training=date_start_training,
            date_end_training=date_end_training,
        )

        training_time = time.time() - t0
        logger.info(
            f"Trained {portfolio_type} portfolio from {date_start_training} to {date_end_training} in {training_time} seconds"
        )

        # testing
        t0 = time.time()

        portfolio_testing = self.test(
            portfolio_type=portfolio_type,
            asset_weights=portfolio_training.asset_weights,
            date_start_testing=date_start_testing,
            date_end_testing=date_end_testing,
        )

        testing_time = time.time() - t0
        logger.info(
            f"Tested portfolio from {date_start_testing} to {date_end_testing} in {testing_time} seconds"
        )

        # preparing the result
        profile_training = self.portfolio_profile(portfolio_training)
        profile_testing = self.portfolio_profile(portfolio_testing)

        profile_training.update(
            {
                "type": "training",
                "date": date_start_training.strftime("%Y/%m/%d"),
                "execution_time": training_time,
            }
        )
        profile_testing.update(
            {
                "type": "testing",
                "date": date_start_testing.strftime("%Y/%m/%d"),
                "execution_time": testing_time,
            }
        )

        result_iter = dict(
            profile_training=profile_training,
            profile_testing=profile_testing,
        )

        return dict(profile_training=profile_training, profile_testing=profile_testing)

    def train(
        self,
        portfolio_type,
        date_start_training,
        date_end_training,
    ):
        data_returns = self.data_returns.loc[date_start_training:date_end_training, :]

        portfolio_training = Portfolio(
            portfolio_type=portfolio_type,
            data_returns=data_returns,
            risk_free_rate=self.risk_free_rate,
            optimization_params=self.optimization_params,
            sampling_params=self.sampling_params,
        )
        portfolio_training.optimize()

        return portfolio_training

    def test(
        self,
        portfolio_type,
        asset_weights,
        date_start_testing,
        date_end_testing,
    ):
        data_returns = self.data_returns.loc[date_start_testing:date_end_testing, :]
        portfolio_testing = Portfolio(
            portfolio_type=portfolio_type,
            data_returns=data_returns,
            asset_weights=asset_weights,
            risk_free_rate=self.risk_free_rate,
            optimization_params=self.optimization_params,
            sampling_params=self.sampling_params,
        )
        return portfolio_testing

    @staticmethod
    def portfolio_profile(portfolio):
        portfolio.evaluate()

        result = dict(
            portfolio_type=portfolio.portfolio_type,
            optimization_params=portfolio.optimization_params,
            sampling_params=portfolio.sampling_params,
            date_start=portfolio.date_start.strftime("%Y/%m/%d"),
            date_end=portfolio.date_end.strftime("%Y/%m/%d"),
            asset_weights=portfolio.asset_weights_dict,
        )
        result.update(portfolio.performance)

        return result


class BackTestAnalysis:
    def __init__(self, backtest, result_path=Path("results") / Path("portfolio")):
        self.backtest = backtest
        self.profile_backtesting = self.backtest.profile_backtesting
        self.result_path = result_path

        self.result_path.mkdir(parents=True, exist_ok=True)

    def plot(self):

        profile_backtesting_test = [
            i["profile_testing"] for i in self.profile_backtesting
        ]
        plot = Plot(path_plot=self.result_path)

        # asset weights
        asset_weight_list = []
        for i in profile_backtesting_test:

            d = i["asset_weights"]
            d.update({"date": i["date"], "portfolio_type": i["portfolio_type"]})
            asset_weight_list.append(d)

        asset_weight_df = (
            pd.DataFrame(asset_weight_list).groupby("portfolio_type").agg("mean")
        )
        asset_weight_df = 100 * asset_weight_df

        plot.plot_bar(
            returns=asset_weight_df,
            title="Average Asset Weights",
            xlabel="Portfolio",
            ylabel="Asset Weights (%)",
            filename="asset_weights",
        )

        # portfolio returns
        returns_df = pd.DataFrame(profile_backtesting_test)[
            ["date", "portfolio_type", "daily_return"]
        ]
        date_list = returns_df["date"].unique()
        portfolio_list = returns_df["portfolio_type"].unique()

        portfolio_returns_list = []

        for date in date_list:

            filt = returns_df["date"] == date
            returns_subset = returns_df.loc[filt, :]

            d = {}
            for p in portfolio_list:
                r = returns_subset.loc[
                    returns_subset["portfolio_type"] == p, "daily_return"
                ]
                d.update({p: float(r)})
            d["date"] = date
            portfolio_returns_list.append(d)

        portfolio_returns_df = pd.DataFrame(portfolio_returns_list).set_index("date")
        portfolio_returns_df = portfolio_returns_df

        plot.plot_box(
            returns=100 * portfolio_returns_df,
            title="Distribution of Daily Returns",
            xlabel="Portfolio",
            ylabel="Daily Returns (%)",
            filename="portfolio_dist",
        )

        plot.plot_trend(
            returns=100 * portfolio_returns_df,
            title="Daily Returns",
            xlabel="Date",
            ylabel="Total Returns (%)",
            filename="portoflio_returns_daily",
        )

        portfolio_returns_cum_df = (portfolio_returns_df + 1).cumprod() - 1
        plot.plot_trend(
            returns=100 * portfolio_returns_cum_df,
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Total Returns (%)",
            filename="portoflio_returns_cum",
        )

        plot.plot_heatmap(
            portfolio_returns_df,
            relation_type="corr",
            title="Portfolio Correlation",
            filename="portfolio_corr",
        )

        plot.plot_heatmap(
            portfolio_returns_df,
            relation_type="cov",
            title="Portfolio Covariance",
            filename="portfolio_cov",
        )

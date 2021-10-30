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
        portfolio_fitness,
        data_returns,
        asset_weights=None,
        risk_free_rate=0.0,
        optimization_params=None,
        sampling_params=None,
    ):
        # args
        self.portfolio_fitness = portfolio_fitness
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
            self.portfolio_fitness,
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


class BackTest:
    def __init__(self, **backtesting_config):
        """[summary]

        Args:
            experiment_name ([type]): [description]
            portfolio_fitness_list ([type]): [description]
            tickers ([type]): [description]
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
        self._backtesting_config = backtesting_config
        self.backtesting_name = backtesting_config["backtesting_name"]
        self.portfolio_fitness_list = backtesting_config["portfolio_fitness_list"]
        self.tickers = backtesting_config["tickers"]
        self.start_date = backtesting_config["start_date"]
        self.end_date = backtesting_config["end_date"]
        self.optimization_params = backtesting_config["optimization_params"]
        self.sampling_params = backtesting_config["sampling_params"]
        self.training_days = backtesting_config["training_days"]
        self.testing_days = backtesting_config["testing_days"]
        self.risk_free_rate = backtesting_config["risk_free_rate"]
        self.n_jobs = backtesting_config["n_jobs"]

        self.asset_list = list(self.tickers.values())

        # create the time windows
        self.analysis_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f"{self.testing_days}D",
        )

        # each window is a tuple of three elements:
        # (the first day of training, the reference day, the last day of testing)
        self.training_delta = pd.Timedelta(self.training_days, unit="d")
        self.testing_delta = pd.Timedelta(self.testing_days, unit="d")
        self.analysis_windows = [
            (i.date() - self.training_delta, i.date(), i.date() + self.testing_delta)
            for i in self.analysis_range
        ]

        # market data
        self.market_data = MarketData(
            asset_list=self.asset_list,
            date_start=self.start_date - pd.Timedelta(self.training_days, unit="d"),
            date_end=self.end_date + pd.Timedelta(self.training_days, unit="d"),
        )

    @property
    def backtesting_config(self):
        return self._backtesting_config

    def get_portfolio_instances(self):
        return [
            dict(
                portfolio_fitness=portfolio_fitness,
                date_start_training=window[0],
                date_end_training=window[1],
                date_start_testing=window[1],
                date_end_testing=window[2],
            )
            for window in self.analysis_windows
            for portfolio_fitness in self.portfolio_fitness_list
        ]

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

    def run_iter(
        self,
        portfolio_fitness,
        date_start_training,
        date_end_training,
        date_start_testing,
        date_end_testing,
    ):
        print(date_start_testing)
        # training
        t0 = time.time()

        portfolio_training = self.train(
            portfolio_fitness=portfolio_fitness,
            date_start_training=date_start_training,
            date_end_training=date_end_training,
        )

        training_time = time.time() - t0
        logger.info(
            f"Trained {portfolio_fitness} portfolio from {date_start_training}"
            f"to {date_end_training} in {training_time} seconds"
        )

        # testing
        t0 = time.time()

        portfolio_testing = self.test(
            portfolio_fitness=portfolio_fitness,
            asset_weights=portfolio_training.asset_weights,
            date_start_testing=date_start_testing,
            date_end_testing=date_end_testing,
        )

        testing_time = time.time() - t0
        logger.info(
            f"Tested portfolio from {date_start_testing} to {date_end_testing}"
            f" in {testing_time} seconds"
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

        return dict(profile_training=profile_training, profile_testing=profile_testing)

    def train(
        self,
        portfolio_fitness,
        date_start_training,
        date_end_training,
    ):
        data_returns = self.market_data.data_returns.loc[
            date_start_training:date_end_training, :
        ]

        portfolio_training = Portfolio(
            portfolio_fitness=portfolio_fitness,
            data_returns=data_returns,
            risk_free_rate=self.risk_free_rate,
            optimization_params=self.optimization_params,
            sampling_params=self.sampling_params,
        )
        portfolio_training.optimize()

        return portfolio_training

    def test(
        self,
        portfolio_fitness,
        asset_weights,
        date_start_testing,
        date_end_testing,
    ):
        data_returns = self.market_data.data_returns.loc[
            date_start_testing:date_end_testing, :
        ]
        portfolio_testing = Portfolio(
            portfolio_fitness=portfolio_fitness,
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
            portfolio_fitness=portfolio.portfolio_fitness,
            optimization_params=portfolio.optimization_params,
            sampling_params=portfolio.sampling_params,
            date_start=portfolio.date_start.strftime("%Y/%m/%d"),
            date_end=portfolio.date_end.strftime("%Y/%m/%d"),
            asset_weights=portfolio.asset_weights_dict,
        )
        result.update(portfolio.performance)

        return result


class BackTestAnalysis:
    def __init__(self, portfolio_backtesting, result_path):
        self.portfolio_backtesting = portfolio_backtesting
        self.profile_backtesting = portfolio_backtesting.profile_backtesting
        self.result_path = result_path
        self.result_path.mkdir(parents=True, exist_ok=True)
        self.profile_backtesting_test = [
            i["profile_testing"] for i in self.profile_backtesting
        ]
        self.plot = Plot(path_plot=self.result_path)
        self.market_returns_plot = MarketDataAnalysis(
            self.portfolio_backtesting.market_data, self.result_path
        )
        # market_returns_plot.plot()

        # portfolio returns
        returns_df = pd.DataFrame(self.profile_backtesting_test)[
            ["date", "portfolio_fitness", "daily_return"]
        ]
        date_list = returns_df["date"].unique()
        portfolio_list = returns_df["portfolio_fitness"].unique()

        portfolio_returns_list = []

        for date in date_list:

            filt = returns_df["date"] == date
            returns_subset = returns_df.loc[filt, :]

            d = {}
            for p in portfolio_list:
                r = returns_subset.loc[
                    returns_subset["portfolio_fitness"] == p, "daily_return"
                ]
                d.update({p: float(r)})
            d["date"] = date
            portfolio_returns_list.append(d)

        self.portfolio_returns_df = pd.DataFrame(portfolio_returns_list).set_index(
            "date"
        )

    def plot_asset_weights(self):
        asset_weight_list = []
        for i in self.profile_backtesting_test:

            d = i["asset_weights"]
            d.update({"date": i["date"], "portfolio_fitness": i["portfolio_fitness"]})
            asset_weight_list.append(d)

        asset_weight_df = (
            pd.DataFrame(asset_weight_list).groupby("portfolio_fitness").agg("mean")
        )
        asset_weight_df = 100 * asset_weight_df

        return self.plot.plot_bar(
            df=asset_weight_df,
            title="Average Asset Weights",
            xlabel="Portfolio Fitness",
            ylabel="Asset Weights (%)",
            filename="portfolio_weights",
        )

    def plot_returns_dist(self):
        return self.plot.plot_box(
            df=100 * self.portfolio_returns_df,
            title="Distribution of Daily Returns",
            xlabel="Portfolio",
            ylabel="Daily Returns (%)",
            filename="portfolio_dist",
        )

    def plot_trends(self):
        portfolio_returns_cum_df = (self.portfolio_returns_df + 1).cumprod() - 1
        return self.plot.plot_trend(
            df=portfolio_returns_cum_df,
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Returns",
            filename="portoflio_returns_cum",
        )

    def plot_corr(self):

        return self.plot.plot_heatmap(
            df=self.portfolio_returns_df,
            relation_type="corr",
            title="Portfolio Correlation",
            annotate=True,
            filename="portfolio_corr",
        )

    def plot_cov(self):

        return self.plot.plot_heatmap(
            df=self.portfolio_returns_df,
            relation_type="cov",
            title="Portfolio Covariance",
            annotate=True,
            filename="portfolio_cov",
        )

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from portfawn.sampling import Sampling
from portfawn.market_data import MarketData
from portfawn.plot import Plot
from portfawn.portfolio_optimization import PortfolioOptimization
from portfawn.utils import get_assets_signature, is_jsonable

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, **portfolio_config):

        # args
        self.portfolio_config = portfolio_config
        self.portfolio_fitness = portfolio_config["portfolio_fitness"]
        self.data_returns = portfolio_config["data_returns"]
        self.optimization_params = portfolio_config.get(
            "optimization_params",
            {
                "scipy_params": {
                    "maxiter": 1000,
                    "disp": False,
                    "ftol": 1e-10,
                },
                "target_return": 0.1,
                "target_risk": 0.1,
                "weight_bound": (0.0, 1.0),
            },
        )
        self.sampling_params = portfolio_config.get(
            "sampling_params", {"type": "standard"}
        )
        self.risk_free_rate = portfolio_config.get("risk_free_rate", 0.0)
        self.asset_weights = portfolio_config.get("asset_weights", None)
        self.annualized_days = 252

        # other params
        self.asset_list = list(self.data_returns.columns)
        self.date_start = self.data_returns.index[0]
        self.date_end = self.data_returns.index[-1]
        self.plot = Plot()

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

        w = self.asset_weights
        returns_np = self.data_returns.to_numpy()
        cov = self.data_returns.cov().to_numpy()

        self.asset_weights_dict = {
            self.asset_list[ind]: float(w) for ind, w in enumerate(self.asset_weights)
        }

        # returns
        portfolio_returns = pd.DataFrame(
            returns_np.dot(w),
            index=self.data_returns.index,
            columns=[self.portfolio_fitness],
        )
        daily_return = portfolio_returns.mean().values[0]
        portfolio_cum_returns = (portfolio_returns + 1).cumprod() - 1
        assets_cum_returns = (self.data_returns + 1).cumprod() - 1
        portfolio_assets_returns = pd.concat(
            [portfolio_returns, self.data_returns], axis=1
        )
        portfolio_assets_cum_returns = pd.concat(
            [portfolio_cum_returns, assets_cum_returns], axis=1
        )

        portfolio_asset_total_return = portfolio_assets_cum_returns.iloc[-1, :]
        portfolio_total_return = portfolio_asset_total_return[self.portfolio_fitness]
        # volatility
        portdolio_sd = np.sqrt(w.T.dot(cov).dot(w))[0][0]

        # market
        market_mean_sd = pd.DataFrame(columns=["mean", "sd"])
        market_mean_sd["mean"] = self.data_returns.mean()
        market_mean_sd["sd"] = self.data_returns.std()

        # portfolio
        portfolio_mean_sd = pd.DataFrame(
            index=[self.portfolio_fitness], columns=["mean", "sd"]
        )
        portfolio_mean_sd["mean"] = daily_return
        portfolio_mean_sd["sd"] = portdolio_sd

        performance = {}
        performance.update(
            {
                "portfolio_returns": portfolio_returns,
                "portfolio_cum_returns": portfolio_cum_returns,
                "portfolio_assets_cum_returns": portfolio_assets_cum_returns,
                "portfolio_assets_returns": portfolio_assets_returns,
                "portfolio_total_return": portfolio_total_return,
                "portfolio_asset_total_return": portfolio_asset_total_return,
                "daily_return": daily_return,
                "daily_sd": portdolio_sd,
                "asset_weights_dict": self.asset_weights_dict,
                "portfolio_config": self.portfolio_config,
                "market_mean_sd": market_mean_sd,
                "portfolio_mean_sd": portfolio_mean_sd,
                "portfolio_asset_mean_sd": pd.concat(
                    [portfolio_mean_sd, market_mean_sd], axis=0
                ),
            }
        )

        self.performance = performance

    def __str__(self):

        if not self.performance:
            raise Exception("The portfolio.evaluate() methos should call first.")

        p = self.performance.copy()
        w_str = json.dumps(p["asset_weights_dict"], sort_keys=True, indent=4)

        out_str = ""
        out_str += f"- asset_weights_dict:\n{w_str}\n\n"
        out_str += f"- daily_return:\n{p['daily_return']}\n\n"
        out_str += f"- daily_sd:\n{p['daily_sd']}\n\n"
        out_str += f"- portfolio_returns:\n{p['portfolio_returns']}\n\n"
        out_str += f"- portfolio_cum_returns:\n{p['portfolio_cum_returns']}\n\n"

        return out_str

    def plot_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.performance["portfolio_returns"],
            title=f"",
            xlabel="Date",
            ylabel="Returns",
            legend=False,
        )
        return fig, ax

    def plot_cum_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.performance["portfolio_assets_cum_returns"],
            title="",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_dist_returns(self):
        fig, ax = self.plot.plot_box(
            df=self.performance["portfolio_assets_returns"],
            title="",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns",
            yscale="symlog",
        )
        return fig, ax

    def plot_corr(self):

        fig, ax = self.plot.plot_heatmap(
            df=self.performance["portfolio_assets_returns"],
            relation_type="corr",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.performance["portfolio_assets_returns"],
            relation_type="cov",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_mean_sd(
        self,
        annualized=True,
        fig=None,
        ax=None,
    ):

        market_mean_sd = self.performance["market_mean_sd"].copy()
        portfolio_mean_sd = self.performance["portfolio_mean_sd"].copy()
        random_mean_sd = self.random_portfolio()

        if annualized:
            market_mean_sd["mean"] *= self.annualized_days
            market_mean_sd["sd"] *= np.sqrt(self.annualized_days)
            portfolio_mean_sd["mean"] *= self.annualized_days
            portfolio_mean_sd["sd"] *= np.sqrt(self.annualized_days)
            random_mean_sd["mean"] *= self.annualized_days
            random_mean_sd["sd"] *= np.sqrt(self.annualized_days)

        fig, ax = self.plot.plot_scatter_portfolio_random(
            df_1=market_mean_sd,
            df_2=portfolio_mean_sd,
            df_3=random_mean_sd,
            title="",
            xlabel="Volatility (SD)",
            ylabel="Expected Returns",
        )

        return fig, ax

    def random_portfolio(self):
        n = 1000
        returns_np = self.data_returns.to_numpy()
        cov = self.data_returns.cov().to_numpy()
        r_list = []
        for i in range(n):
            w_rand = np.random.random((1, cov.shape[0]))
            w_rand = w_rand / w_rand.sum()
            r = returns_np.dot(w_rand.T).mean()
            c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0]
            r_list.append({"mean": r, "sd": c})
        return pd.DataFrame(r_list)


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
        self.portfolio_fitness = backtesting_config["portfolio_fitness"]
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
        self.tickers_inv = {v: k for k, v in self.tickers.items()}
        self.portfolio_fitness_list = list(self.portfolio_fitness.keys())
        self.annualized_days = 252

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
            tickers=self.tickers,
            date_start=self.start_date - pd.Timedelta(self.training_days, unit="d"),
            date_end=self.end_date + pd.Timedelta(self.training_days, unit="d"),
        )

        self.plot = Plot()

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

        # create profiles
        profile_backtesting = profile_backtesting
        data_list = []

        for profile in profile_backtesting:
            for mode in ["profile_training", "profile_testing"]:

                curr = profile[mode]
                d = {
                    "type": curr["type"],
                    "portfolio_fitness": curr["portfolio_fitness"],
                    "date_start": curr["date_start"],
                    "date_end": curr["date_end"],
                    "date": curr["date"],
                    "portfolio_daily_return": curr["daily_return"],
                    "portfolio_daily_sd": curr["daily_sd"],
                    "portfolio_total_return": curr["portfolio_total_return"],
                    "portfolio_asset_total_return": curr[
                        "portfolio_asset_total_return"
                    ].to_dict(),
                    "portfolio_asset_mean_sd": curr[
                        "portfolio_asset_mean_sd"
                    ].to_dict(),
                    "portfolio_mean_sd": curr["portfolio_mean_sd"],
                    "asset_weights_dict": curr["asset_weights_dict"],
                    "execution_time": curr["execution_time"],
                    "optimization_params": curr["optimization_params"],
                    "sampling_params": curr["sampling_params"],
                    "mode": mode,
                }

                data_list.append(d)

        profile_df = pd.DataFrame(data_list)

        portfolio_df = profile_df.loc[profile_df["type"] == "testing", :].set_index(
            "date", inplace=False
        )
        portfolio_returns_df = pd.DataFrame()

        for portfolio_fitness in portfolio_df["portfolio_fitness"].unique():
            temp = portfolio_df.loc[
                portfolio_df["portfolio_fitness"] == portfolio_fitness, :
            ]
            portfolio_returns_df[portfolio_fitness] = temp["portfolio_total_return"]

        portfolio_cum_returns_df = (portfolio_returns_df + 1).cumprod() - 1

        self.profile_backtesting = profile_backtesting
        self.profile_df = profile_df
        self.portfolio_returns_df = portfolio_returns_df
        self.portfolio_cum_returns_df = portfolio_cum_returns_df
        self.asset_weights_df = pd.DataFrame(
            [item for ind, item in profile_df["asset_weights_dict"].items()],
            index=profile_df["date"],
        )
        self.mean_sd = pd.concat([i for i in profile_df["portfolio_mean_sd"]])

    def run_iter(
        self,
        portfolio_fitness,
        date_start_training,
        date_end_training,
        date_start_testing,
        date_end_testing,
    ):
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

    def plot_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.portfolio_returns_df,
            title="",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_cum_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.portfolio_cum_returns_df,
            title="",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_dist_returns(self):

        fig, ax = self.plot.plot_box(
            df=self.portfolio_returns_df,
            title="",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns (%)",
        )
        return fig, ax

    def plot_corr(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.portfolio_returns_df,
            relation_type="corr",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.portfolio_returns_df,
            relation_type="cov",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_asset_weights(self):
        fig, ax = self.plot.plot_trend(
            df=self.asset_weights_df,
            title="",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_asset_weights_dist(self):
        fig, ax = self.plot.plot_box(
            df=self.asset_weights_df,
            title="",
            xlabel="Date",
            ylabel="Cumulative Returns",
            yscale="symlog",
        )
        return fig, ax

    def plot_mean_sd(self, annualized=True):

        mean_sd = self.mean_sd.copy()

        if annualized:
            mean_sd["mean"] *= self.annualized_days
            mean_sd["sd"] *= np.sqrt(self.annualized_days)

        fig, ax = self.plot.plot_scatter_seaborn(
            data=mean_sd,
            y="mean",
            x="sd",
            hue=mean_sd.index,
            title="",
            xlabel="Volatility (SD)",
            ylabel="Expected Returns",
        )
        return fig, ax


class MultiPortoflio:
    def __init__(self, **portfolio_config):

        # args
        self.portfolio_config = portfolio_config
        self.portfolio_fitness_list = portfolio_config["portfolio_fitness_list"]
        self.data_returns = portfolio_config["data_returns"]
        self.optimization_params = portfolio_config.get(
            "optimization_params",
            {
                "scipy_params": {
                    "maxiter": 1000,
                    "disp": False,
                    "ftol": 1e-10,
                },
                "target_return": 0.1,
                "target_risk": 0.1,
                "weight_bound": (0.0, 1.0),
            },
        )
        self.sampling_params = portfolio_config.get(
            "sampling_params", {"type": "standard"}
        )
        self.risk_free_rate = portfolio_config.get("risk_free_rate", 0.0)

        self.annualized_days = 252
        self.plot = Plot()

    def generate(self):

        # optimized portfolios
        mean_sd_list = []

        for portfolio_fitness in self.portfolio_fitness_list:
            portfolio = Portfolio(
                portfolio_fitness=portfolio_fitness,
                data_returns=self.data_returns,
                risk_free_rate=self.risk_free_rate,
                optimization_params=self.optimization_params,
                sampling_params=self.sampling_params,
            )
            portfolio.optimize()
            portfolio.evaluate()
            mean_sd_list.append(portfolio.performance["portfolio_mean_sd"])

        market_mean_sd = portfolio.performance["market_mean_sd"]
        portfolio_mean_sd = pd.concat(mean_sd_list, axis=0)

        market_mean_sd["mean"] *= self.annualized_days
        market_mean_sd["sd"] *= np.sqrt(self.annualized_days)
        portfolio_mean_sd["mean"] *= self.annualized_days
        portfolio_mean_sd["sd"] *= np.sqrt(self.annualized_days)

        # random portfolios
        n = 1000
        returns_np = portfolio.data_returns.to_numpy()
        cov = portfolio.data_returns.cov().to_numpy()
        r_list = []
        for i in range(n):
            w_rand = np.random.random((1, cov.shape[0]))
            w_rand = w_rand / w_rand.sum()
            r = returns_np.dot(w_rand.T).mean() * self.annualized_days
            c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0] * np.sqrt(
                self.annualized_days
            )
            r_list.append({"mean": r, "sd": c})
        mean_sd_random = pd.DataFrame(r_list)

        self.market_mean_sd = market_mean_sd
        self.portfolio_mean_sd = portfolio_mean_sd
        self.mean_sd_random = mean_sd_random

    def plot_portfolio(self):
        fig, ax = self.plot.plot_scatter_portfolio_random(
            df_1=self.market_mean_sd,
            df_2=self.portfolio_mean_sd,
            df_3=self.mean_sd_random,
            title="Expected Returns vs. Volatility",
            xlabel="Volatility (SD)",
            ylabel="Expected Returns",
        )
        return fig, ax

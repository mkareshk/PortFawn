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

        # volatility
        portdolio_std = np.sqrt(w.T.dot(cov).dot(w))[0][0]

        # market
        market_mean_std = pd.DataFrame(columns=["mean", "std"])
        market_mean_std["mean"] = self.data_returns.mean()
        market_mean_std["std"] = self.data_returns.std()

        # portfolio
        portfolio_mean_std = pd.DataFrame(
            index=[self.portfolio_fitness], columns=["mean", "std"]
        )
        portfolio_mean_std["mean"] = daily_return
        portfolio_mean_std["std"] = portdolio_std

        performance = {}
        performance.update(
            {
                "portfolio_returns": portfolio_returns,
                "portfolio_cum_returns": portfolio_cum_returns,
                "portfolio_assets_returns": portfolio_assets_returns,
                "portfolio_asset_total_return": portfolio_asset_total_return,
                "daily_return": daily_return,
                "daily_std": portdolio_std,
                "asset_weights_dict": self.asset_weights_dict,
                "portfolio_config": self.portfolio_config,
                "portfolio_asset_mean_std": pd.concat(
                    [portfolio_mean_std, market_mean_std], axis=0
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
        out_str += f"- daily_std:\n{p['daily_std']}\n\n"
        out_str += f"- portfolio_returns:\n{p['portfolio_returns']}\n\n"
        out_str += f"- portfolio_cum_returns:\n{p['portfolio_cum_returns']}\n\n"

        return out_str

    def plot_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.performance["portfolio_returns"],
            title=f"{self.portfolio_fitness} Returns",
            xlabel="Date",
            ylabel="Returns",
            legend=False,
        )
        return fig, ax

    def plot_cum_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.performance["portfolio_assets_cum_returns"],
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_dist_returns(self):
        fig, ax = self.plot.plot_box(
            df=100 * self.performance["portfolio_assets_returns"],
            title="Distribution of Daily Returns",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns (%)",
        )
        return fig, ax

    def plot_corr(self):

        fig, ax = self.plot.plot_heatmap(
            df=self.performance["portfolio_assets_returns"],
            relation_type="corr",
            title="Portfolio Correlation",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.performance["portfolio_assets_returns"],
            relation_type="cov",
            title="Portfolio Covariance",
            annotate=True,
        )
        return fig, ax

    def plot_mean_std(
        self,
        annualized=True,
        fig=None,
        ax=None,
    ):

        market_mean_std = self.performance["market_mean_std"]
        portfolio_mean_std = self.performance["portfolio_mean_std"]
        random_mean_std = self.random_portfolio()

        if annualized:
            market_mean_std["mean"] *= self.annualized_days
            market_mean_std["std"] *= np.sqrt(self.annualized_days)
            portfolio_mean_std["mean"] *= self.annualized_days
            portfolio_mean_std["std"] *= np.sqrt(self.annualized_days)
            random_mean_std["mean"] *= self.annualized_days
            random_mean_std["std"] *= np.sqrt(self.annualized_days)

        fig, ax = self.plot.plot_scatter_portfolio_random(
            df_1=market_mean_std,
            df_2=portfolio_mean_std,
            df_3=random_mean_std,
            title="Expected Returns vs. Volatility",
            xlabel="Volatility (STD)",
            ylabel="Expected Returns",
        )

        return fig, ax

    def random_portfolio(self):
        n = 100000
        returns_np = self.data_returns.to_numpy()
        cov = self.data_returns.cov().to_numpy()
        r_list = []
        for i in range(n):
            w_rand = np.random.random((1, cov.shape[0]))
            w_rand = w_rand / w_rand.sum()
            r = returns_np.dot(w_rand.T).mean()
            c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0]
            r_list.append({"mean": r, "std": c})
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
        self.backtesting_config = portfolio_backtesting.backtesting_config
        self.result_path = result_path
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.profile_backtesting_test = [
            i["profile_training"]
            for i in self.profile_backtesting  # TODO: testing or training
        ]
        self.plot = Plot()

        # market_returns_plot.plot()

        # portfolio returns
        returns_df = pd.DataFrame(self.profile_backtesting_test)[
            ["date", "portfolio_fitness", "daily_return", "daily_std"]
        ]
        date_list = returns_df["date"].unique()
        portfolio_list = returns_df["portfolio_fitness"].unique()

        portfolio_returns_list = []
        portfolio_risk_list = []

        for date in date_list:
            d_return = {"date": date}
            d_risk = {"date": date}
            for p in portfolio_list:
                filt = (returns_df["date"] == date) & (
                    returns_df["portfolio_fitness"] == p
                )
                r, s = returns_df.loc[filt, ["daily_return", "daily_std"]].values[0]
                d_return.update({f"{p}": float(r)})
                d_risk.update({f"{p}": float(s)})
            portfolio_returns_list.append(d_return)
            portfolio_risk_list.append(d_risk)

        self.portfolio_returns = pd.DataFrame(portfolio_returns_list).set_index("date")
        self.portfolio_risk = pd.DataFrame(portfolio_risk_list).set_index("date")

        self.portfolio_returns.columns = [
            self.portfolio_backtesting.portfolio_fitness[i]
            for i in self.portfolio_returns.columns
        ]
        self.portfolio_risk.columns = [
            self.portfolio_backtesting.portfolio_fitness[i]
            for i in self.portfolio_risk.columns
        ]
        self._mean_std = pd.DataFrame(columns=["mean", "std"])
        self._mean_std["mean"] = self.portfolio_returns.mean()
        self._mean_std["std"] = self.portfolio_risk.mean()
        print(self._mean_std)

    def plot_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.portfolio_returns,
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_cum_returns(self):
        portfolio_returns_cum_df = (self.portfolio_returns + 1).cumprod() - 1
        fig, ax = self.plot.plot_trend(
            df=portfolio_returns_cum_df,
            title="Cumulative Returns",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_dist_returns(self):

        fig, ax = self.plot.plot_box(
            df=100 * self.portfolio_returns,
            title="Distribution of Daily Returns",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns (%)",
        )
        return fig, ax

    def plot_corr(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.portfolio_returns,
            relation_type="corr",
            title="Portfolio Correlation",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.portfolio_returns,
            relation_type="cov",
            title="Portfolio Covariance",
            annotate=True,
        )
        return fig, ax

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

        asset_weight_df.index = [
            self.portfolio_backtesting.portfolio_fitness[i]
            for i in asset_weight_df.index
        ]

        fig, ax = self.plot.plot_bar(
            df=asset_weight_df,
            title="Average Asset Weights",
            xlabel="Portfolio Fitness",
            ylabel="Asset Weights (%)",
        )
        return fig, ax

    def plot_mean_std(
        self,
        annualized=True,
        fig=None,
        ax=None,
    ):

        ms = self._mean_std.copy()

        if annualized:
            ms["mean"] *= 252
            ms["std"] *= np.sqrt(252)

        fig, ax = self.plot.plot_scatter_portfolio(
            df_1=ms,
            df_2=self.portfolio_backtesting.market_data.mean_std,
            title="Expected Returns vs. Volatility",
            xlabel="Volatility (STD)",
            ylabel="Expected Returns",
        )
        return fig, ax

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

        asset_weight_df.index = [
            self.portfolio_backtesting.portfolio_fitness[i]
            for i in asset_weight_df.index
        ]

        fig, ax = self.plot.plot_bar(
            df=asset_weight_df,
            title="Average Asset Weights",
            xlabel="Portfolio Fitness",
            ylabel="Asset Weights (%)",
        )
        return fig, ax

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from portfawn.expected_stats import ExpectedStats
from portfawn.market_data import MarketData, PlotMarketData
from portfawn.plot import Plot
from portfawn.portfolio_optimization import PortfolioOptimization
from portfawn.utils import get_assets_signature, is_jsonable

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, portfolio_name, data_returns, return_type, sample_type, asset_weights=None, risk_free_rate=0.0):
        # args
        self.portfolio_name = portfolio_name
        self.data_returns = data_returns
        self.return_type = return_type
        self.sample_type = sample_type
        self.asset_weights = asset_weights
        self.risk_free_rate = risk_free_rate

        # other params
        self.asset_list = list(data_returns.columns)
        self.date_start = data_returns.iloc[0]['date']
        self.date_end = data_returns.iloc[-1]['date']

        # self.freq_dict = get_freq_list()
        # self.freq_name = self.freq_dict[self.return_type]

    def optimize(self):
        if self.asset_weights:
            raise Exception(f"The portfolio weights have already set, {self.asset_weights}")

        expected_stats = ExpectedStats(data_returns=self.data_returns, type='simple')
        expected_return = expected_stats.expected_return
        expected_risk = expected_stats.expected_risk
        optimizer = PortfolioOptimization(expected_return=expected_return, expected_risk=expected_risk,
                                          risk_free_rate=self.risk_free_rate)

        self.weights = optimizer.optimize(optimization_type='simple')

        return {self.asset_list[ind]: float(w) for ind, w in enumerate(self.weights)}

    def evaluate(self):
        w = self.weights

        performance = {}

        performance.update({f"{self.portfolio_name}_daily_return": self.data_returns.mean().dot(w)})

        return performance


class PortfolioBackTesting:

    def __init__(
            self, asset_list, portfolio_names, start_date_analysis, end_date_analysis, training_days, testing_days,
            return_type="D", sample_type="full", risk_free_rate=0.0, path_data=Path("data"),
            path_results=Path("results")):
        # parameters
        self.asset_list = asset_list
        self.portfolio_names = portfolio_names
        self.start_date_analysis = start_date_analysis
        self.end_date_analysis = end_date_analysis
        self.training_days = training_days
        self.testing_days = testing_days
        self.return_type = return_type
        self.sample_type = sample_type
        self.risk_free_rate = risk_free_rate
        self.path_data = path_data
        self.path_results = path_results

        # creating the time windows
        self.analysis_range = pd.date_range(start=self.start_date_analysis,
                                            end=self.end_date_analysis,
                                            freq=f"{self.testing_days}D")

        # each window: (the first day of training, the reference day, the last day of testing)
        self.training_delta = pd.Timedelta(self.training_days, unit="d")
        self.testing_delta = pd.Timedelta(self.testing_days, unit="d")
        self.analysis_windows = [(i.date() - self.training_delta, i.date(), i.date() + self.testing_delta)
                                 for i in self.analysis_range]

        # market data
        self.market_data = MarketData(asset_list=self.asset_list, date_start=self.start_date_analysis,
                                      date_end=self.end_date_analysis)
        self.data_returns = self.market_data.data_returns

    def run(self):
        return [self.run_iter(**instance) for instance in self.get_portfolio_instances()]

    def get_portfolio_instances(self):
        return [dict(portfolio_name=portfolio_name, date_start_training=window[0], date_end_training=window[1],
                     date_start_testing=window[1], date_end_testing=window[2])
                for window in self.analysis_windows
                for portfolio_name in self.portfolio_names]

    def run_iter(self, portfolio_name, date_start_training, date_end_training, date_start_testing, date_end_testing):
        profile_training = self.train(portfolio_name, date_start_training, date_end_training)
        logger.info(f"Trained {portfolio_name} portfolio from {date_start_training} to {date_end_training}")

        profile_testing = self.test(portfolio_name, profile_training["asset_weights"], date_start_testing,
                                    date_end_testing)
        logger.info(f"Tested portfolio from {date_start_testing} to {date_end_testing}")

        return dict(profile_training=profile_training, profile_testing=profile_testing)

    def train(self, portfolio_name, date_start_training, date_end_training):
        data_returns = self.data_returns.loc[date_start_training:date_end_training, :]
        portfolio_training = Portfolio(portfolio_name=portfolio_name, data_returns=data_returns,
                                       return_type=self.return_type, sample_type=self.sample_type,
                                       risk_free_rate=self.risk_free_rate)
        portfolio_training.optimize()

        return self.portfolio_profile(portfolio_training)

    def test(self, portfolio_name, asset_weights, date_start_testing, date_end_testing):
        data_returns = self.data_returns.loc[date_start_testing:date_end_testing, :]
        portfolio_testing = Portfolio(portfolio_name=portfolio_name, data_returns=data_returns,
                                      asset_weights=asset_weights,
                                      return_type=self.return_type, sample_type=self.return_type,
                                      risk_free_rate=self.risk_free_rate)
        return self.portfolio_profile(portfolio_testing)

    @staticmethod
    def portfolio_profile(portfolio):
        portfolio.evaluate()

        result = dict(portfolio_name=portfolio.portfolio_name, date_start=portfolio.date_start,
                      date_end=portfolio.data_end, asset_weights=portfolio.asset_weights)
        result.update(portfolio.asset_performance)

        return result

    def plot(self):
        tr_te = f"tr{self.training_days}_te{self.testing_days}"

        portfolio_analysis_plot = Plot(
            asset_num=len(self.asset_list),
            path_results=self.path_results,
            plot_type="portfolio_analysis",
        )

        # trends
        portfolio_analysis_plot.plot_trend(
            returns=(1 + self.result_returns).cumprod() - 1,
            title="Cumulative Returns of Different Portfolio Types",
            xlabel="Date",
            ylabel="Cumulative Returns",
            filename=f"portfolio_analysis_cum_returns_{tr_te}",
        )

        portfolio_analysis_plot.plot_trend(
            returns=self.result_sharpe_ratio,
            title="Sharpe Ratio for Different Portfolio Types",
            xlabel="Date",
            ylabel="Sharpe Ratio ($S_p$)",
            filename=f"portfolio_analysis_sharpe_ratio_{tr_te}",
        )

        portfolio_analysis_plot.plot_box(
            self.result_returns.loc[
            :, self.result_returns.columns.str.contains("portfolio")
            ].rename(
                columns=lambda x: x.replace("portfolio_total_return_", "")
                    .replace("_", " ")
                    .capitalize()
            ),
            title="Distribution of Total Returns for Different Portfolio Types",
            xlabel="Portfolio Types",
            ylabel="Returns",
            filename=f"portfolio_analysis_returns_{tr_te}",
        )

        portfolio_analysis_plot.plot_bar(
            self.asset_weights_agg_df,
            yscale="log",
            title="Average of Asset Weights in Different Portfolio Types",
            legend_title="Assets",
            xlabel="Portfolio Types",
            ylabel="Asset Weight ($w$)",
            filename="portfolio_asset_weights",
        )


class PlotPortfolio:
    def __init__(self, portfolio, path_data=Path("data"), path_results=Path("results")):
        self.portfolio = portfolio
        self.path_data, self.path_results = self.create_path(path_data, path_results)

        path_data.mkdir(parents=True, exist_ok=True)
        path_results.mkdir(parents=True, exist_ok=True)

        # logging
        self.logger = logging.getLogger(__name__)

        PlotMarketData(self.portfolio.market_data, path_results=self.path_results)
        self.store_results()
        summary = self.portfolio_summary().replace("    ", " ").replace("\n", "")
        self.logger.info(f"The summary of the portfolio: {summary}")

    def plot_all(self):
        self.store_results()
        self.plot_figs()
        self.store_csvs()
        self.portfolio_summary()

    def store_results(self):
        self.plot = Plot(
            asset_num=len(self.portfolio.asset_list),
            path_results=self.path_results,
            plot_type="portfolio",
        )
        self.plot_figs()
        self.store_csvs()
        self.portfolio_summary()

    def plot_figs(self):
        f_name = self.portfolio.freq_name
        f_name_cap = f_name.capitalize()
        p_type = self.portfolio.name
        p_type_cap = p_type.capitalize()
        returns = (
            self.portfolio.performance["portfolio_asset_daily_return"]
                .resample("M")
                .mean()
                .pct_change()
        )
        cum = self.portfolio.performance["portfolio_asset_daily_cum"]
        # box
        corr_wo_diag_df = returns.corr()
        np.fill_diagonal(corr_wo_diag_df.values, 0.0)
        self.plot.plot_box(
            returns=returns,
            title=f"Distribution of {p_type_cap} Portfolio {f_name_cap} Returns",
            xlabel="Assets",
            ylabel=f"{f_name_cap} Returns",
            filename=f"portfolio_box_{f_name}_returns_{p_type}_{self.portfolio.portfolio_sig}",
        )
        # heatmap
        self.plot.plot_heatmap(
            returns.corr(),
            "corr",
            f"Correlation of {f_name_cap} Returns for {p_type_cap} Portfolio",
            f"portfolio_corr_{f_name}_returns_{p_type}_{self.portfolio.portfolio_sig}",
        )
        self.plot.plot_heatmap(
            returns.cov(),
            "cov",
            f"Covariance of {f_name_cap} Returns for {p_type_cap} Portfolio",
            f"portfolio_cov_{f_name}_returns_{p_type}_{self.portfolio.portfolio_sig}",
        )
        # trends
        self.plot.plot_trend(
            returns=returns,
            title=f"Trends of {p_type_cap} Portfolio {f_name_cap} Returns",
            xlabel="Assets",
            ylabel=f"{f_name_cap} Returns",
            filename=f"portfolio_trend_{f_name}_returns_{p_type}_{self.portfolio.portfolio_sig}",
        )
        self.plot.plot_trend(
            returns=cum,
            title=f"{p_type_cap} Portfolio {f_name_cap} Cumulative Returns",
            xlabel="Assets",
            ylabel=f"{f_name_cap} Returns",
            filename=f"portfolio_trend_cum_{f_name}_returns_{p_type}_{self.portfolio.portfolio_sig}",
        )

    def store_csvs(self):
        f_name = self.portfolio.freq_name
        p_type = self.portfolio.name
        path = self.path_results / Path("returns")
        self.path_results.mkdir(parents=True, exist_ok=True)
        returns = self.portfolio.performance["portfolio_asset_daily_return"]
        cum = self.portfolio.performance["portfolio_asset_daily_cum"]
        returns.to_csv(
            path
            / f"portfolio_returns_{f_name}_{p_type}_{self.portfolio.portfolio_sig}.csv"
        )
        cum.to_csv(
            path / f"portfolio_cum_{f_name}_{p_type}_{self.portfolio.portfolio_sig}.csv"
        )
        filename = path / Path(
            f"portfolio_stats_{p_type}_{self.portfolio.portfolio_sig}.csv"
        )
        returns.describe().to_csv(filename)

    def portfolio_summary(self):
        portfolio_summary = {}
        portfolio_summary.update({"name": self.portfolio.name})
        portfolio_summary.update({"asset_list": self.portfolio.asset_list})
        portfolio_summary.update({"weights": self.portfolio.weights.tolist()})
        portfolio_summary.update(
            {"date_start": self.portfolio.date_start.strftime("%Y-%m-%d")}
        )
        portfolio_summary.update(
            {"date_end": self.portfolio.date_end.strftime("%Y-%m-%d")}
        )
        portfolio_summary.update({"risk_free_rate": self.portfolio.risk_free_rate})
        for k, v in self.portfolio.performance.items():
            if is_jsonable(v):
                portfolio_summary.update({k: v})
        summary_str = json.dumps(portfolio_summary, indent=4)
        filename = self.path_results / Path(
            f"summary_portfolio_{self.portfolio.portfolio_sig}.txt"
        )
        with open(filename, "wt") as fout:
            fout.write(summary_str)
        return summary_str

    def create_path(self, path_data, path_results):
        self.weights_hash = hashlib.md5(
            "".join([str(i) for i in self.portfolio.asset_weights]).encode("utf-8")
        ).hexdigest()[0:5]
        self.portfolio.portfolio_sig = f"{self.weights_hash}"
        self.portfolio.market_data_sig = get_assets_signature(
            self.portfolio.asset_list,
            self.portfolio.date_start,
            self.portfolio.date_end,
        )
        path_data = path_data / Path(self.portfolio.market_data_sig)
        path_results = path_results / Path(self.portfolio.market_data_sig)
        return path_data, path_results

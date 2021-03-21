import json
import time
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from portfawn.market_data import MarketData, PlotMarketData
from portfawn.portfolio_optimization import PortfolioOptimization
from portfawn.plot import Plot
from portfawn.expected_stats import ExpectedStats
from portfawn.utils import get_freq_list, get_assets_signature, is_jsonable


class Portfolio:
    def __init__(
        self,
        name,
        asset_list,
        asset_weights,
        return_type,
        sample_type,
        date_start,
        date_end,
        risk_free_rate,
    ):

        # parameters
        self.name = name
        self.asset_list = asset_list
        self.asset_weights = asset_weights
        self.return_type = return_type
        self.sample_type = sample_type
        self.date_start = date_start
        self.date_end = date_end
        self.risk_free_rate = risk_free_rate
        self.freq_dict = get_freq_list()
        self.freq_name = self.freq_dict[self.return_type]

        # random seed
        np.random.seed(int(time.time()))

        # log
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Start creating the {self.name} portfolio ({self.return_type}) using {len(self.asset_list)} "
            f"assets with the data between {self.date_start.strftime('%Y-%m-%d')} and {self.date_end.strftime('%Y-%m-%d')}"
        )

        # market data
        self.market_data = MarketData(
            asset_list=self.asset_list,
            date_start=self.date_start,
            date_end=self.date_end,
        )
        self.logger.info(
            f"The data of {self.name} portfolio ({self.return_type}) is collected successfully"
        )

        # optimization
        if isinstance(self.asset_weights, str):  # train mode

            # claculate the expected returns and risks
            expected_stats = ExpectedStats(
                returns_data=self.market_data.get_data(
                    freq=self.return_type, metric="returns"
                ),
                optimization_type=self.name,
            )
            mean, cov = expected_stats.expected_mean_cov(
                sample_type=self.sample_type, instance_num=10
            )

            # optimization
            optimizer = PortfolioOptimization(
                returns_mean=mean,
                returns_cov=cov,
                risk_free_rate=self.risk_free_rate,
            )
            self.weights = optimizer.optimize(optimization_type=self.asset_weights)
            self.logger.info(
                f"{self.name} portfolio ({self.return_type}) is trained by "
                f"{self.asset_weights}, weights are: {self.weights.tolist()}"
            )

        elif isinstance(self.asset_weights, np.ndarray):  # test mode
            self.weights = self.asset_weights.reshape(-1, 1)
            self.logger.info(
                f"{self.name} portfolio is tested using weights {self.weights}"
            )

        # portfolio performance
        self.calc_performance()

    def calc_performance(self):
        w = self.weights

        diff = self.date_end - self.date_start
        years_num = diff.days / 365.25
        days_in_year_mean = self.market_data.business_day_num / years_num

        self.performance = {}

        # daily returns (assets, portfolio)
        self.performance.update(
            {
                "portfolio_asset_daily_return": pd.concat(
                    [
                        self.market_data.get_data(freq="D", metric="returns"),
                        self.market_data.get_data(freq="D", metric="returns").dot(w)[0],
                    ],
                    axis=1,
                )
            }
        )
        self.performance["portfolio_asset_daily_return"].columns = [
            *self.performance["portfolio_asset_daily_return"].columns[:-1],
            self.name,
        ]

        # cumulative returns (assets, portfolio)
        self.performance.update(
            {
                "portfolio_asset_daily_cum": (
                    self.performance["portfolio_asset_daily_return"] + 1
                ).cumprod()
                - 1
            }
        )

        # mean returns (assets, portfolio)
        self.performance.update(
            {
                "portfolio_asset_daily_mean": self.performance[
                    "portfolio_asset_daily_return"
                ].mean()
            }
        )

        # std returns (assets, portfolio)
        self.performance.update(
            {
                "portfolio_asset_daily_std": self.performance[
                    "portfolio_asset_daily_return"
                ].std()
            }
        )

        # total returns (assets, portfolio)
        self.performance.update(
            {
                "portfolio_asset_total_return": pd.concat(
                    [
                        self.market_data.get_data(freq="D", metric="total_return"),
                        self.market_data.get_data(freq="D", metric="total_return").dot(
                            w
                        ),
                    ],
                    axis=1,
                )
            }
        )
        self.performance["portfolio_asset_total_return"].columns = [
            *self.performance["portfolio_asset_total_return"].columns[:-1],
            self.name,
        ]

        # annual returns (portfolio)
        self.performance.update(
            {
                "annual_portfolio_asset_return": self.performance[
                    "portfolio_asset_total_return"
                ]
                / years_num
            }
        )

        # annual std (portfolio)
        self.performance.update(
            {
                "annual_portfolio_asset_std": self.performance[
                    "portfolio_asset_daily_std"
                ][self.name]
                * np.sqrt(days_in_year_mean)
            }
        )

        # sharpe ratio (assets, portfolio)
        self.performance.update(
            {
                "sharpe_ratio": (
                    self.performance["annual_portfolio_asset_return"]
                    - self.risk_free_rate
                )
                / (self.performance["annual_portfolio_asset_std"])
            }
        )

    @property
    def frequency(self):
        if self.return_type == "D":
            return 252
        elif self.return_type == "W":
            return 52
        elif self.return_type == "M":
            return 12
        elif self.return_type == "Q":
            return 4
        elif self.return_type == "Y":
            return 1
        else:
            raise Exception


class PortfolioBackTesting:
    def __init__(
        self,
        asset_list,
        portfolio_names,
        start_date_analysis,
        end_date_analysis,
        training_days,
        testing_days,
        risk_free_rate,
        path_data=Path("data"),
        path_results=Path("results"),
    ):

        # parameters
        self.asset_list = asset_list
        self.portfolio_names = portfolio_names
        self.start_date_analysis = start_date_analysis
        self.end_date_analysis = end_date_analysis
        self.training_days = training_days
        self.testing_days = testing_days
        self.risk_free_rate = risk_free_rate
        self.path_data = path_data
        self.path_results = path_results

        # creating the time windows
        self.analysis_range = pd.date_range(
            start=self.start_date_analysis,
            end=self.end_date_analysis,
            freq=f"{self.testing_days}D",
        )
        # each window: (the first day of training, today, the last day of testing)
        self.training_delta = pd.Timedelta(self.training_days, unit="d")
        self.testing_delta = pd.Timedelta(self.testing_days, unit="d")
        self.analysis_windows = [
            (i.date() - self.training_delta, i.date(), i.date() + self.testing_delta)
            for i in self.analysis_range
        ]

    def analyze(self):

        results_list = []
        asset_weight_list = []
        for window in self.analysis_windows:

            result = {}
            result.update({"date": window[1]})
            print(self.training_days, self.testing_days, window[1])

            for portfolio_name in self.portfolio_names:

                # train
                trained_portfolio = Portfolio(
                    name=portfolio_name,
                    asset_list=self.asset_list,
                    asset_weights=portfolio_name,
                    return_type="D",
                    sample_type="full",
                    date_start=window[0],
                    date_end=window[1],
                    risk_free_rate=self.risk_free_rate,
                )

                # store asset weights
                asset_weight_dict = {
                    self.asset_list[ind]: float(w)
                    for ind, w in enumerate(trained_portfolio.weights)
                }
                asset_weight_dict.update(
                    {"date": window[1], "portfolio_name": portfolio_name}
                )
                asset_weight_list.append(asset_weight_dict)

                # test
                test_portfolio = Portfolio(
                    name=portfolio_name,
                    asset_list=self.asset_list,
                    asset_weights=trained_portfolio.weights,
                    return_type="D",
                    sample_type=None,
                    date_start=window[1],
                    date_end=window[2],
                    risk_free_rate=self.risk_free_rate,
                )

                # plot
                # plot_portfolio = PlotPortfolio(test_portfolio)
                # plot_portfolio.plot_all()

                # portfolio results
                result.update(
                    {
                        f"portfolio_total_return_{portfolio_name}": test_portfolio.performance[
                            "portfolio_asset_total_return"
                        ][
                            portfolio_name
                        ].values[
                            0
                        ]
                    }
                )
                result.update(
                    {
                        f"portfolio_sharpe_ratio_{portfolio_name}": test_portfolio.performance[
                            "sharpe_ratio"
                        ][
                            portfolio_name
                        ].values[
                            0
                        ]
                    }
                )

            # asset results
            for a in self.asset_list:
                result.update(
                    {
                        f"asset_total_return_{a}": test_portfolio.performance[
                            "portfolio_asset_total_return"
                        ][a].values[0]
                    }
                )
                result.update(
                    {
                        f"asset_sharpe_ratio_{a}": test_portfolio.performance[
                            "sharpe_ratio"
                        ][a].values[0]
                    }
                )

            results_list.append(result)

        # store the results
        result_df = pd.DataFrame(results_list)
        result_df.set_index("date", inplace=True)
        self.result_sharpe_ratio = result_df.loc[
            :, result_df.columns.str.contains("_sharpe_ratio_")
        ]
        self.result_returns = result_df.loc[
            :, result_df.columns.str.contains("_total_return_")
        ]
        self.result_df = result_df
        self.result_df.to_csv(self.path_results / Path("portfolio_results.csv"))

        # store asset weights
        self.asset_weights_df = pd.DataFrame(asset_weight_list)
        self.asset_weights_df.to_csv(
            self.path_results / Path("portfolio_asset_weights_raw.csv")
        )
        self.asset_weights_agg_df = self.asset_weights_df.groupby("portfolio_name").agg(
            "mean"
        )
        self.asset_weights_agg_df.to_csv(
            self.path_results / Path("portfolio_asset_weights_aggregated.csv")
        )

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

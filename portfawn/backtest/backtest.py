import time
import logging

import pandas as pd
from joblib import Parallel, delayed
from portfawn.plot import Plot


class BackTest:
    plot = Plot()
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        portfolio_list,
        asset_list,
        date_start,
        date_end,
        fitting_days,
        evaluation_days,
        n_jobs,
    ):

        self.portfolio_list = portfolio_list
        self.asset_list = asset_list
        self.date_start = date_start
        self.date_end = date_end
        self.fitting_days = fitting_days
        self.evaluation_days = evaluation_days
        self.n_jobs = n_jobs

        # create the time windows
        self.analysis_range = pd.date_range(
            start=self.date_start,
            end=self.date_end,
            freq=f"{self.evaluation_days}D",
        )

        # each window is a tuple of three elements:
        # (the first day of training, the reference day, the last day of testing)
        self.fitting_delta = pd.Timedelta(self.fitting_days, unit="d")
        self.evaluation_delta = pd.Timedelta(self.evaluation_days, unit="d")
        self.analysis_windows = [
            (i.date() - self.fitting_delta, i.date(), i.date() + self.evaluation_delta)
            for i in self.analysis_range
        ]

    def run(self):

        backtesting_instances = [
            dict(
                portfolio=portfolio,
                date_start_training=window[0],
                date_end_training=window[1],
                date_start_testing=window[1],
                date_end_testing=window[2],
            )
            for window in self.analysis_windows
            for portfolio in self.portfolio_list
        ]

        # sequential
        if self.n_jobs == 1:
            performance_backtesting = [
                self.run_iter(**instance) for instance in backtesting_instances
            ]

        # parallel
        elif self.n_jobs > 1:
            performance_backtesting = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_iter)(**instance) for instance in backtesting_instances
            )

        # create profiles
        data_list = []

        for performance in performance_backtesting:
            d = {
                "portfolio_total_return": performance["portfolio_total_return"].values[
                    0
                ],
                "portfolio_asset_total_return": performance[
                    "portfolio_asset_total_return"
                ].to_dict(),
                "portfolio_asset_mean_sd": performance[
                    "portfolio_asset_mean_sd"
                ].to_dict(),
                "portfolio_mean_sd": performance["portfolio_mean_sd"],
                "asset_weights": performance["asset_weights"],
                "fitting_time": performance["fitting_time"],
                "evaluation_time": performance["evaluation_time"],
                "date": performance["date"],
            }

            d.update(performance["portfolio_config"])
            data_list.append(d)

        profile_df = pd.DataFrame(data_list)

        portfolio_df = profile_df.set_index("date", inplace=False)
        portfolio_returns_df = pd.DataFrame()

        for portfolio_name in portfolio_df["name"].unique():
            temp = portfolio_df.loc[portfolio_df["name"] == portfolio_name, :]
            portfolio_returns_df[portfolio_name] = temp["portfolio_total_return"]

        portfolio_cum_returns_df = (portfolio_returns_df + 1).cumprod() - 1

        self.profile_backtesting = performance_backtesting
        self.profile_df = profile_df
        self.portfolio_returns_df = portfolio_returns_df
        self.portfolio_cum_returns_df = portfolio_cum_returns_df
        self.asset_weights_df = pd.DataFrame(
            [item for ind, item in profile_df["asset_weights"].items()],
            index=profile_df["date"],
        )
        self.mean_sd = pd.concat([i for i in profile_df["portfolio_mean_sd"]])

    def run_iter(
        self,
        portfolio,
        date_start_training,
        date_end_training,
        date_start_testing,
        date_end_testing,
    ):

        # training
        t0 = time.time()

        portfolio.fit(
            asset_list=self.asset_list,
            date_start=date_start_training,
            date_end=date_end_training,
        )

        fitting_time = time.time() - t0
        self.logger.info(
            f"Trained {portfolio.name} portfolio from {date_start_training}"
            f" to {date_end_training} in {fitting_time} seconds"
        )

        # testing
        t0 = time.time()

        performance = portfolio.evaluate(
            date_start=date_start_testing, date_end=date_end_testing
        )

        evaluation_time = time.time() - t0
        self.logger.info(
            f"Tested {portfolio.name} portoflio from {date_start_testing} to {date_end_testing}"
            f" in {evaluation_time} seconds"
        )

        performance.update(
            {
                "fitting_time": fitting_time,
                "evaluation_time": evaluation_time,
                "date": date_start_testing,
            }
        )

        return performance

    def plot_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.portfolio_cum_returns_df,
            title=f"",
            xlabel="Date",
            ylabel="Returns",
            legend=False,
        )
        return fig, ax

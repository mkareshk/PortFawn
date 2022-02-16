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

        # performance

        # returns
        total_returns_list = [p["total_returns"] for p in performance_backtesting]
        rolling_total_returns = pd.concat(total_returns_list, axis=1).T
        self.returns = rolling_total_returns.groupby(
            by=rolling_total_returns.index
        ).max()

        # cumulative returns
        self.cum_returns = (self.returns + 1).cumprod() - 1

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
            df=self.returns,
            title="",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_cum_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.cum_returns,
            title="",
            xlabel="Date",
            ylabel="Returns",
        )
        return fig, ax

    def plot_dist_returns(self):

        fig, ax = self.plot.plot_box(
            df=self.returns,
            title="",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns (%)",
        )
        return fig, ax

    def plot_corr(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.returns,
            relation_type="corr",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.returns,
            relation_type="cov",
            title="",
            annotate=True,
        )
        return fig, ax

    # def plot_asset_weights(self):
    #     fig, ax = self.plot.plot_trend(
    #         df=self.asset_weights_df,
    #         title="",
    #         xlabel="Date",
    #         ylabel="Returns",
    #     )
    #     return fig, ax

    # def plot_asset_weights_dist(self):
    #     fig, ax = self.plot.plot_box(
    #         df=self.asset_weights_df,
    #         title="",
    #         xlabel="Date",
    #         ylabel="Cumulative Returns",
    #         yscale="symlog",
    #     )
    #     return fig, ax

    # def plot_mean_sd(self, annualized=True):

    #     mean_sd = self.mean_sd.copy()

    #     if annualized:
    #         mean_sd["mean"] *= self.annualized_days
    #         mean_sd["sd"] *= np.sqrt(self.annualized_days)

    #     fig, ax = self.plot.plot_scatter_seaborn(
    #         data=mean_sd,
    #         y="mean",
    #         x="sd",
    #         hue=mean_sd.index,
    #         title="",
    #         xlabel="Volatility (SD)",
    #         ylabel="Expected Returns",
    #     )
    #     return fig, ax

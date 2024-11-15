import logging
import time

import dafin
import pandas as pd
from joblib import Parallel, delayed

from ..plot import Plot

logger = logging.getLogger(__name__)


class BackTest:
    """
    A class to perform backtesting of financial portfolios.

    Attributes:
        plot (Plot): An instance of the `Plot` class for generating visualizations.
        logger (logging.Logger): Logger for logging messages and errors.
        portfolio_list (list): List of portfolio objects to backtest.
        asset_list (list): List of asset identifiers to analyze.
        date_start (str): The start date for the backtesting period (YYYY-MM-DD).
        date_end (str): The end date for the backtesting period (YYYY-MM-DD).
        fitting_days (int): Number of days used for training the portfolio.
        evaluation_days (int): Number of days used for evaluating the portfolio.
        n_jobs (int): Number of parallel jobs to run (1 for sequential processing).
        analysis_range (DatetimeIndex): Dates defining the evaluation windows.
        analysis_windows (list): List of tuples defining training and testing periods.
        asset_returns (ReturnsData): Data object for retrieving asset returns.
        returns (DataFrame): Average daily returns of portfolios.
        cum_returns (DataFrame): Cumulative returns of portfolios.
    """

    plot = Plot()

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
        """
        Initializes the BackTest object with the given parameters.

        Args:
            portfolio_list (list): List of portfolio objects to evaluate.
            asset_list (list): List of asset identifiers for analysis.
            date_start (str): Start date for backtesting (YYYY-MM-DD).
            date_end (str): End date for backtesting (YYYY-MM-DD).
            fitting_days (int): Days for training the portfolio model.
            evaluation_days (int): Days for evaluating portfolio performance.
            n_jobs (int): Number of parallel jobs to run (1 for sequential).
        """

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

        # retrieve asset returns
        self.asset_returns = dafin.ReturnsData(self.asset_list)

    def run(self):
        """
        Executes the backtesting process.

        Generates performance metrics for each portfolio over the defined
        analysis windows, either sequentially or in parallel.

        Returns:
            None
        """

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
        total_returns_list = [p["returns_total"] for p in performance_backtesting]

        rolling_total_returns = pd.concat(total_returns_list, axis=1).T
        self.returns = rolling_total_returns.groupby(rolling_total_returns.index).mean()
        self.returns.sort_index(inplace=True)

        self.cum_returns = (self.returns + 1).cumprod() - 1

    def run_iter(
        self,
        portfolio,
        date_start_training,
        date_end_training,
        date_start_testing,
        date_end_testing,
    ):
        """
        Executes a single iteration of the backtesting process for a portfolio.

        Args:
            portfolio: The portfolio object to backtest.
            date_start_training (datetime.date): Start date of the training period.
            date_end_training (datetime.date): End date of the training period.
            date_start_testing (datetime.date): Start date of the testing period.
            date_end_testing (datetime.date): End date of the testing period.

        Returns:
            dict: Performance metrics for the backtested portfolio.
        """

        training_returns = self.asset_returns.get_returns(
            date_start=date_start_training, date_end=date_end_training
        )
        evaluation_returns = self.asset_returns.get_returns(
            date_start=date_start_testing, date_end=date_end_testing
        )
        # training
        t0 = time.time()

        try:
            portfolio.fit(training_returns)
        except Exception as e:
            logger.error(f"Error training portfolio {portfolio.name}: {e}")
            return None

        fitting_time = time.time() - t0

        logger.info(
            f"Trained {portfolio.name} portfolio from {date_start_training} "
            f"to {date_end_training} in {fitting_time:.2f} seconds"
        )

        # evaluation
        t0 = time.time()
        returns_total = portfolio.evaluate(evaluation_returns).returns_total
        evaluation_time = time.time() - t0

        logger.info(
            f"Tested {portfolio.name} portfolio from {date_start_testing} to {date_end_testing}"
            f" in {evaluation_time} seconds"
        )

        # portfolio performance
        performance = {
            "fitting_time": fitting_time,
            "evaluation_time": evaluation_time,
            "date": date_start_testing,
            "returns_total": returns_total,
        }

        return performance

    def plot_returns(self):
        """
        Plots average daily returns of portfolios.

        Returns:
            tuple: Figure and axis objects for the plot.
        """

        fig, ax = self.plot.plot_trend(
            df=self.returns,
            title="Average Daily Returns of Portfolios",
            xlabel="Date",
            ylabel="Returns",
            portfolio_list=[p.name for p in self.portfolio_list],
        )
        return fig, ax

    def plot_cum_returns(self):
        """
        Plots cumulative returns of portfolios.

        Returns:
            tuple: Figure and axis objects for the plot.
        """
        fig, ax = self.plot.plot_trend(
            df=self.cum_returns,
            title="Cumulative Returns of Portfolios",
            xlabel="Date",
            ylabel="Returns",
            portfolio_list=[p.name for p in self.portfolio_list],
        )
        return fig, ax

    def plot_dist_returns(self):
        """
        Plots a boxplot of the distribution of daily returns.

        Returns:
            tuple: Figure and axis objects for the plot.
        """
        fig, ax = self.plot.plot_box(
            df=self.returns,
            title="Distribution of Daily Returns",
            xlabel="Portfolio",
            ylabel="Daily Returns (%)",
        )
        return fig, ax

    def plot_corr(self):
        """
        Plots a heatmap of return correlations between portfolios.

        Returns:
            tuple: Figure and axis objects for the plot.
        """
        fig, ax = self.plot.plot_heatmap(
            df=self.returns,
            relation_type="corr",
            title="Correlations Between Portfolios",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        """
        Plots a heatmap of return covariances between portfolios.

        Returns:
            tuple: Figure and axis objects for the plot.
        """

        fig, ax = self.plot.plot_heatmap(
            df=self.returns,
            relation_type="cov",
            title="Covariances Between pPortfolios",
            annotate=True,
        )
        return fig, ax

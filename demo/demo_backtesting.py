import logging

import dafin
import matplotlib.pyplot as plt

from portfawn import (
    BackTest,
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    OptimizationModel,
    RandomPortfolio,
)

# Configure logging
logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] (%(name)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to showcase backtesting using the PortFawn package.
    """
    try:
        # Parameters
        logger.info("Initializing parameters...")
        asset_list = ["SPY", "BND", "GLD"]  # Example assets
        date_start = "2010-01-01"
        date_end = "2022-12-30"
        fitting_days = 252  # 1 year
        evaluation_days = 22  # 1 month
        n_jobs = 12  # Parallel jobs for backtesting

        logger.info(
            "Assets: %s, Start Date: %s, End Date: %s, Fitting Days: %d, Evaluation Days: %d",
            asset_list,
            date_start,
            date_end,
            fitting_days,
            evaluation_days,
        )

        # Portfolio initialization
        logger.info("Initializing portfolio models...")
        portfolio_names = {
            "MVP": "Minimum Variance Portfolio",
            "MRP": "Maximum Return Portfolio",
            "MSRP": "Maximum Sharpe Ratio Portfolio",
            "BMOP": "Binary Multi-Objective Portfolio",
        }

        mean_variance_portfolios = [
            MeanVariancePortfolio(
                name=portfolio_names[o],
                optimization_model=OptimizationModel(objective=o),
            )
            for o in ["MRP", "MVP", "MSRP", "BMOP"]
        ]
        portfolio_list = [
            RandomPortfolio(),
            EquallyWeightedPortfolio(),
        ]
        portfolio_list.extend(mean_variance_portfolios)

        logger.info("Total portfolios initialized: %d", len(portfolio_list))
        for i, portfolio in enumerate(portfolio_list, start=1):
            logger.info("Portfolio %d: %s", i, portfolio.name)

        # Backtesting setup
        logger.info("Setting up backtesting...")
        backtest = BackTest(
            portfolio_list=portfolio_list,
            asset_list=asset_list,
            date_start=date_start,
            date_end=date_end,
            fitting_days=fitting_days,
            evaluation_days=evaluation_days,
            n_jobs=n_jobs,
        )

        # Run backtesting
        logger.info("Running backtesting...")
        backtest.run()
        logger.info("Backtesting completed successfully.")

        # Visualization
        logger.info("Generating and saving plots...")
        save_plot(backtest.plot_returns(), "plot_returns.png", "Returns Plot")
        save_plot(
            backtest.plot_cum_returns(),
            "plot_cum_returns.png",
            "Cumulative Returns Plot",
        )
        save_plot(
            backtest.plot_dist_returns(),
            "plot_dist_returns.png",
            "Return Distributions Plot",
        )
        save_plot(backtest.plot_corr(), "plot_corr.png", "Correlation Plot")
        save_plot(backtest.plot_cov(), "plot_cov.png", "Covariance Plot")

        logger.info("All plots saved successfully.")

    except Exception as e:
        logger.critical("Critical error in main execution: %s", str(e), exc_info=True)


def save_plot(fig_ax, filename, plot_name):
    """
    Save a plot to a file.

    Args:
        fig_ax (tuple): A tuple containing the figure and axes of the plot.
        filename (str): The filename to save the plot to.
        plot_name (str): A descriptive name for the plot (used in logs).
    """
    try:
        fig, ax = fig_ax
        fig.savefig(filename)
        logger.info("%s saved as %s.", plot_name, filename)
    except Exception as e:
        logger.error("Failed to save %s: %s", plot_name, str(e), exc_info=True)


if __name__ == "__main__":
    logger.info("Starting PortFawn backtesting showcase...")
    main()
    logger.info("Finished PortFawn backtesting showcase.")

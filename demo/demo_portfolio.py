import logging

import dafin

from portfawn import (
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
    Main function to showcase portfolio optimization using the PortFawn package.
    """
    try:
        # Parameters
        logger.info("Initializing parameters...")
        asset_list = ["SPY", "BND", "GLD"]  # Example assets
        date_start = "2010-01-01"
        date_end = "2022-12-30"

        logger.info(
            "Assets: %s, Start Date: %s, End Date: %s",
            asset_list,
            date_start,
            date_end,
        )

        # Load data
        logger.info("Loading returns data...")
        data_instance = dafin.ReturnsData(asset_list)
        returns_data = data_instance.get_returns(date_start, date_end)

        logger.info("Successfully loaded returns data for %d assets.", len(asset_list))

        # Portfolio optimization setup
        logger.info("Initializing portfolio optimization models...")
        portfolio_names = {
            "MVP": "Minimum Variance Portfolio",
            "MRP": "Maximum Return Portfolio",
            "MSRP": "Maximum Sharpe Ratio Portfolio",
            "BMOP": "Binary Multi-Objective Portfolio (BMOP)",
        }
        mean_variance_portfolios = [
            MeanVariancePortfolio(
                name=portfolio_names[o],
                optimization_model=OptimizationModel(objective=o),
            )
            for o in ["MRP", "MSRP", "BMOP"]
        ]
        portfolio_list = [
            RandomPortfolio(),
            EquallyWeightedPortfolio(),
        ]
        portfolio_list.extend(mean_variance_portfolios)

        logger.info("Total portfolios initialized: %d", len(portfolio_list))

        # Fit and evaluate portfolios
        for i, portfolio in enumerate(portfolio_list, start=1):
            try:
                logger.info(
                    "Fitting portfolio %d/%d: %s",
                    i,
                    len(portfolio_list),
                    portfolio.__class__.__name__,
                )
                portfolio.fit(returns_data)

                logger.info("Evaluating portfolio %d/%d...", i, len(portfolio_list))
                performance = portfolio.evaluate(returns_data)

                logger.info(
                    "Portfolio %s performance: %s",
                    portfolio.__class__.__name__,
                    performance,
                )
                print(f"Portfolio: {portfolio}")
                print(f"Performance: {performance}")
                print("-" * 50)

            except Exception as e:
                logger.error(
                    "Error processing portfolio %d: %s", i, str(e), exc_info=True
                )

    except Exception as e:
        logger.critical("Critical error in main execution: %s", str(e), exc_info=True)


if __name__ == "__main__":
    logger.info("Starting PortFawn portfolio optimization showcase...")
    main()
    logger.info("Finished PortFawn portfolio optimization showcase.")

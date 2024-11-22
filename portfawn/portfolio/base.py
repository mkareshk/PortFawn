import logging
from typing import Optional

import numpy as np
import pandas as pd
from dafin import Performance, ReturnsData

from ..models import MeanVarianceRiskModel, OptimizationModel

logger = logging.getLogger(__name__)


class PortfolioBase:

    def __init__(
        self,
        name: str = None,
        risk_model: Optional[MeanVarianceRiskModel] = None,
        optimization_model: Optional[OptimizationModel] = None,
        rf_asset: str = "BND",
        benchmark_asset: str = "SPY",
    ) -> None:
        """
        Initializes the PortfolioBase with the given parameters.

        Args:
            name (str, optional): Name of the portfolio.
            risk_model (MeanVarianceRiskModel, optional): The risk model to use for the portfolio.
            optimization_model (OptimizationModel, optional): The optimization model to use for the portfolio.
            rf_asset (str, optional): The symbol of the risk-free asset. Defaults to "BND".
            benchmark_asset (str, optional): The symbol of the benchmark asset. Defaults to "SPY".
        """

        self.name = name
        self.risk_model = risk_model
        self.optimization_model = optimization_model
        self.rf_asset = rf_asset
        self.benchmark_asset = benchmark_asset

        self.data_rf = ReturnsData(assets=self.rf_asset)
        self.data_benchmark = ReturnsData(assets=self.benchmark_asset)

    def evaluate(self, returns_assets: pd.DataFrame) -> Performance:
        """
        Evaluates the portfolio performance.

        Args:
            returns_assets (pd.DataFrame): DataFrame containing the returns of assets.

        Returns:
            Performance: The performance metrics of the portfolio.

        Raises:
            ValueError: If the portfolio is not properly fitted or if the input data is inconsistent
                with the portfolio's setup.
        """

        # Check if asset list and asset weights are initialized
        if not (hasattr(self, "asset_list") and hasattr(self, "asset_weights")):
            msg = "Fit the portfolio before evaluation."
            logger.error(msg)
            raise ValueError(msg)

        # Ensure consistency between asset list and asset weights
        if len(self.asset_list) != len(self.asset_weights):
            msg = f"Asset list ({self.asset_list}) and asset weights ({self.asset_weights}) are inconsistent."
            logger.error(msg)
            raise ValueError(msg)

        # Ensure consistency between asset weights dictionary and numpy format
        if self._w.shape[0] != len(self.asset_weights):
            msg = f"Asset weights dictionary ({self.asset_weights}) and asset weights in numpy format ({self._w}) are inconsistent."
            logger.error(msg)
            raise ValueError(msg)

        # Ensure fitted asset weights and asset returns to evaluate are consistent
        if set(returns_assets.columns) != set(self.asset_list):
            msg = f"Fitted asset weights ({self.asset_weights}) and asset returns to evaluate are inconsistent."
            logger.error(msg)
            raise ValueError(msg)

        # Convert portfolio returns to DataFrame
        returns_portfolio = pd.DataFrame(
            returns_assets.to_numpy().dot(self._w),
            index=returns_assets.index,
            columns=[self.name],
        )
        date_start = returns_portfolio.index[0]
        date_end = returns_portfolio.index[-1]

        logger.debug(f"Evaluating portfolio from {date_start} to {date_end}")

        returns_rf: pd.DataFrame = self.data_rf.get_returns(
            date_start=date_start, date_end=date_end
        )
        returns_benchmark: pd.DataFrame = self.data_benchmark.get_returns(
            date_start=date_start, date_end=date_end
        )

        self._performance: Performance = Performance(
            returns_assets=returns_portfolio,
            returns_rf=returns_rf,
            returns_benchmark=returns_benchmark,
        )

        return self._performance

    def store_returns_var(self, returns_assets: pd.DataFrame) -> None:
        """
        Stores essential details from the provided asset returns DataFrame.

        Args:
            returns_assets (pd.DataFrame): DataFrame containing the returns of assets.
        """
        self.asset_list = list(returns_assets.columns)
        self.date_start = returns_assets.index[0]
        self.date_end = returns_assets.index[-1]

        logger.debug(f"Asset list: {self.asset_list}")
        logger.debug(f"Date range from {self.date_start} to {self.date_end}")

    def __str__(self) -> str:
        """
        Returns a string representation of the portfolio, summarizing its key attributes.

        Returns:
            str: Summary of the portfolio's attributes.
        """

        summary = f"Portfolio: {self.name}\n"
        if self.risk_model:
            summary += f"\t - Risk Model: {self.risk_model}\n"
        if self.optimization_model:
            summary += f"\t - Optimization Model: {self.optimization_model.objective}\n"
        if hasattr(self, "asset_list"):
            summary += f"\t - Asset List: {self.asset_list}\n"
        else:
            summary += "\t - Asset List: Not set.\n"
        if hasattr(self, "asset_weights"):
            summary += f"\t - Asset Weights: {self.asset_weights}\n"
        else:
            summary += "\t - Asset Weights: Not set.\n"
        if hasattr(self, "date_start"):
            summary += f"\t - Start Date: {self.date_start}\n"
        else:
            summary += "\t - Start Date: Not set.\n"
        if hasattr(self, "date_end"):
            summary += f"\t - End Date: {self.date_end}\n"
        else:
            summary += "\t - End Date: Not set.\n"
        if hasattr(self, "_performance"):
            summary += f"\t - Performance:\n{self._performance.summary}\n"
        else:
            summary += "\t - Performance: Not evaluated yet.\n"

        return summary

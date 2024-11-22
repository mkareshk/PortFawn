import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..models import MeanVarianceRiskModel, OptimizationModel
from .base import PortfolioBase

logger = logging.getLogger(__name__)


class MeanVariancePortfolio(PortfolioBase):

    def __init__(
        self,
        name: str = "mean_variance_portfolio",
        risk_model: Optional[MeanVarianceRiskModel] = None,
        optimization_model: Optional[OptimizationModel] = None,
        rf_asset: str = "BND",
        benchmark_asset: str = "SPY",
    ) -> None:
        """
        Initializes the MeanVariancePortfolio with the given parameters.

        Args:
            name (str, optional): Name of the portfolio. Defaults to "mean_variance_portfolio".
            risk_model (MeanVarianceRiskModel, optional): The risk model used to evaluate the
                asset returns. If None, a default `MeanVarianceRiskModel` instance is created.
            optimization_model (OptimizationModel, optional): The optimization model used to
                calculate optimal weights. If None, a default `OptimizationModel` instance is
                created with "MSRP" as the objective.
            rf_asset (str, optional): The symbol representing the risk-free asset. Defaults to "BND".
            benchmark_asset (str, optional): The symbol representing the benchmark asset. Defaults to "SPY".
        """

        if risk_model is None:
            risk_model = MeanVarianceRiskModel()

        if optimization_model is None:
            optimization_model = OptimizationModel(objective="MSRP")

        super().__init__(
            name, risk_model, optimization_model, rf_asset, benchmark_asset
        )

    def fit(self, returns_assets: pd.DataFrame) -> "MeanVariancePortfolio":
        """
        Fits the portfolio by evaluating risk and optimizing weights.

        This method evaluates the risk of the provided asset returns using the
        specified risk model and then calculates optimal weights using the optimization
        model. The resulting weights are normalized to sum to 1.

        Args:
            returns_assets (pd.DataFrame): A DataFrame where rows represent time periods
                and columns represent asset returns.

        Returns:
            MeanVariancePortfolio: The instance of the `MeanVariancePortfolio` class.

        Raises:
            ValueError: If optimization fails or if the resulting weights do not sum to 1.
        """

        # Store essential details from the asset returns
        self.store_returns_var(returns_assets)

        # Evaluate the risk associated with the assets
        linear_biases, quadratic_biases = self.risk_model.evaluate(returns_assets)
        logger.debug("Risk evaluation completed.")

        try:
            # Optimize the asset weights using the provided biases
            self._w = self.optimization_model.optimize(linear_biases, quadratic_biases)
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise ValueError(f"Optimization failed: {e}")

        # Validate weights
        if not np.isclose(np.sum(self._w), 1):
            msg = "Optimized weights do not sum to 1."
            logger.error(msg)
            raise ValueError(msg)

        # Store the asset weights as a dictionary
        self.asset_weights = dict(zip(self.asset_list, self._w))

        logger.debug(f"Optimized weights: {self.asset_weights}")

        return self

import pandas as pd
from dafin import ReturnsData

from .base import PortfolioBase
from ..models import RiskModel, OptimizationModel


class MeanVariancePortfolio(PortfolioBase):
    """
    Represents a portfolio optimized based on the Mean-Variance optimization approach.

    Inherits from:
    --------------
    PortfolioBase : Base class for portfolios.

    Example:
    --------
    >>> # Using mock classes and methods for demonstration
    ... class MockRiskModel:
    ...     def evaluate(self, returns_assets):
    ...         return ([1, 2], [3, 4])
    ...
    ... class MockOptimizationModel:
    ...     def optimize(self, linear_biases, quadratic_biases):
    ...         return [0.5, 0.5]
    ...
    ... portfolio = MeanVariancePortfolio("MVP", MockRiskModel(), MockOptimizationModel())
    ... returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [-0.01, 0.03]})
    ... portfolio.fit(returns)
    <...MeanVariancePortfolio object at ...>
    >>> portfolio.asset_weights
    {'A': 0.5, 'B': 0.5}
    """

    def __init__(
        self,
        name: str,
        risk_model: "RiskModel",
        optimization_model: "OptimizationModel",
    ) -> None:
        """
        Initializes the MeanVariancePortfolio with the given name, risk model, and optimization model.

        Parameters:
        -----------
        name : str
            Name of the portfolio.
        risk_model : RiskModel
            Model used to assess the risk associated with the portfolio.
        optimization_model : OptimizationModel
            Model used to optimize the portfolio based on mean-variance criteria.
        """
        super().__init__(name, risk_model, optimization_model)

    def fit(self, returns_assets: ReturnsData) -> "MeanVariancePortfolio":
        """
        Fits the portfolio by evaluating the risk and then optimizing using the provided risk and optimization models.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        MeanVariancePortfolio
            Returns the instance of the class.
        """

        # Store essential details from the asset returns
        self.store_returns_var(returns_assets)

        # Evaluate the risk associated with the assets
        linear_biases, quadratic_biases = self.risk_model.evaluate(returns_assets)

        # Optimize the asset weights using the provided biases
        self._w = self.optimization_model.optimize(linear_biases, quadratic_biases)

        # Store the asset weights as a dictionary
        self.asset_weights = dict(zip(self.asset_list, self._w))

        return self

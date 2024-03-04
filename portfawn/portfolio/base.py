import dafin
import pandas as pd

from ..models import RiskModel, OptimizationModel


class PortfolioBase:
    """
    Base class for portfolios, providing fundamental structure and attributes.

    Attributes:
    -----------
    name : str
        Name of the portfolio.
    risk_model : RiskModel, optional
        Model used to assess the risk associated with the portfolio.
    optimization_model : OptimizationModel, optional
        Model used to optimize the portfolio.

    Example:
    --------
    >>> portfolio = PortfolioBase(name="Sample Portfolio")
    >>> portfolio.name
    'Sample Portfolio'
    >>> portfolio.risk_model is None
    True
    >>> portfolio.optimization_model is None
    True
    """

    def __init__(
        self,
        name: str,
        risk_model: "RiskModel" = None,
        optimization_model: "OptimizationModel" = None,
    ) -> None:
        """
        Initializes the PortfolioBase with a name, and optionally with a risk and optimization model.

        Parameters:
        -----------
        name : str
            Name of the portfolio.
        risk_model : RiskModel, optional
            Model used to assess risk. Defaults to None.
        optimization_model : OptimizationModel, optional
            Model used for optimization. Defaults to None.
        """
        self.name = name
        self.risk_model = risk_model
        self.optimization_model = optimization_model

    def evaluate(self, returns_assets: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the portfolio returns based on the provided asset returns.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the portfolio returns.

        Raises:
        -------
        ValueError
            If there are inconsistencies between assets and weights.

        Example:
        --------
        >>> # Mocking instance variables for demonstration
        ... self = type('', (), {})()
        ... self.asset_list = ['A', 'B']
        ... self.asset_weights = {'A': 0.5, 'B': 0.5}
        ... self._w = np.array([0.5, 0.5])
        ... self.name = "Sample Portfolio"
        ... returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [-0.01, 0.03]})
        >>> evaluate(self, returns)
            Sample Portfolio
        0                0.0
        1                0.025
        """

        # Check if asset list and asset weights are initialized
        if not (self.asset_list and self.asset_weights):
            raise ValueError("Fit the portfolio before evaluation.")

        # Ensure consistency between asset list and asset weights
        if len(self.asset_list) != len(self.asset_weights):
            raise ValueError(
                f"Asset list ({self.asset_list}) and asset weights ({self.asset_weights}) are inconsistent."
            )

        # Ensure consistency between asset weights dictionary and numpy format
        if self._w.shape[0] != len(self.asset_weights):
            raise ValueError(
                f"Asset weights dictionary ({self.asset_weights}) and asset weights in numpy format ({self._w}) are inconsistent."
            )

        # Ensure fitted asset weights and asset returns to evaluate are consistent
        if set(returns_assets.columns) != set(self.asset_list):
            raise ValueError(
                f"Fitted asset weights ({self.asset_weights}) and asset returns to evaluate are inconsistent."
            )

        # Calculate portfolio returns
        returns_portfolio_np = returns_assets.to_numpy().dot(self._w)

        # Convert portfolio returns to DataFrame
        returns_portfolio = pd.DataFrame(
            returns_portfolio_np,
            index=returns_assets.index,
            columns=[self.name],
        )

        self._performance = dafin.Performance(returns_assets=returns_portfolio)

        return self._performance

    def store_returns_var(self, returns_assets: pd.DataFrame) -> None:
        """
        Stores essential details from the provided asset returns DataFrame.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            DataFrame containing the returns of assets.

        Example:
        --------
        >>> # Mocking instance variables for demonstration
        ... self = type('', (), {})()
        ... returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [-0.01, 0.03]}, index=["2023-01-01", "2023-01-02"])
        >>> store_returns_var(self, returns)
        >>> self.asset_list
        ['A', 'B']
        >>> self.date_start
        '2023-01-01'
        >>> self.date_end
        '2023-01-02'
        """

        # Store column names, start date, and end date from the provided DataFrame
        self.asset_list = list(returns_assets.columns)
        self.date_start = returns_assets.index[0]
        self.date_end = returns_assets.index[-1]

    def __str__(self) -> str:

        summary = f"Portfolio: {self.name}\n"
        # summary += f"\t - Risk Model: {self.risk_model}\n"
        # summary += f"\t - Optimization Model: {self.optimization_model.objective}\n"
        summary += f"\t - Asset List: {self.asset_list}\n"
        summary += f"\t - Asset Weights: {self.asset_weights}\n"
        summary += f"\t - Start Date: {self.date_start}\n"
        summary += f"\t - End Date: {self.date_end}\n"
        summary += f"\t - Performance:\n{self._performance.summary}\n"

        return summary

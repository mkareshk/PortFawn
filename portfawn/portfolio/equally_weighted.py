import numpy as np
import pandas as pd

from .base import PortfolioBase


class EquallyWeightedPortfolio(PortfolioBase):
    """
    Represents a portfolio where all assets are equally weighted.

    Inherits from:
    --------------
    PortfolioBase : Base class for portfolios.

    Example:
    --------
    >>> portfolio = EquallyWeightedPortfolio("Equally Weighted")
    >>> returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [-0.01, 0.03]})
    >>> portfolio.fit(returns)
    >>> portfolio.asset_weights
    {'A': 0.5, 'B': 0.5}
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the EquallyWeightedPortfolio with the given name.

        Parameters:
        -----------
        name : str
            Name of the portfolio.
        """
        super().__init__(name=name)

    def fit(self, returns_assets: pd.DataFrame) -> None:
        """
        Fits the portfolio by setting equal weights to each asset.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            DataFrame containing the returns of assets.
        """

        # Store essential details from the asset returns
        self.store_returns_var(returns_assets)

        # Calculate equal weights for assets
        num_assets = len(self.asset_list)
        self._w = np.ones(num_assets) / num_assets

        # Store the asset weights as a dictionary
        self.asset_weights = dict(zip(self.asset_list, self._w))

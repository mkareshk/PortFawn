import numpy as np
import pandas as pd

from .base import PortfolioBase


class RandomPortfolio(PortfolioBase):
    """
    Represents a portfolio where assets are assigned random weights.

    Inherits from:
    --------------
    PortfolioBase : Base class for portfolios.

    Example:
    --------
    >>> portfolio = RandomPortfolio("Random Weighted")
    >>> returns = pd.DataFrame({'A': [0.01, 0.02], 'B': [-0.01, 0.03]})
    >>> portfolio.fit(returns)
    <...RandomPortfolio object at ...>
    >>> sum(portfolio.asset_weights.values())
    1.0
    """

    def __init__(self, name: str = "random_portfolio") -> None:
        """
        Initializes the RandomPortfolio with the given name.

        Parameters:
        -----------
        name : str
            Name of the portfolio.
        """
        super().__init__(name)

    def fit(self, returns_assets: pd.DataFrame) -> "RandomPortfolio":
        """
        Fits the portfolio by assigning random weights to each asset.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        RandomPortfolio
            Returns the instance of the class.
        """

        # Store essential details from the asset returns
        self.store_returns_var(returns_assets)

        # Generate random weights for assets
        w = np.random.uniform(low=0.0, high=1.0, size=len(self.asset_list))
        self._w = w / np.sum(w)

        # Store the asset weights as a dictionary
        self.asset_weights = dict(zip(self.asset_list, self._w))

        return self

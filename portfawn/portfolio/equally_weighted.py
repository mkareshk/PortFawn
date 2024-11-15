import logging

import numpy as np
import pandas as pd

from .base import PortfolioBase

logger = logging.getLogger(__name__)


class EquallyWeightedPortfolio(PortfolioBase):
    """
    Represents a portfolio where each asset is assigned an equal weight.

    This class provides a simple portfolio allocation method that assigns equal weights
    to all assets in the portfolio, ensuring the total weight sums to 1.

    Attributes:
    -----------
    asset_list : list
        List of asset names available in the portfolio.
    asset_weights : dict
        Dictionary of assets and their corresponding equal weights.
    _w : np.ndarray
        Array of equal weights assigned to assets.
    """

    def __init__(self, name: str = "equally_weighted_portfolio") -> None:
        """
        Initializes the EquallyWeightedPortfolio with a given name.

        Parameters:
        -----------
        name : str, optional
            Name of the portfolio. Default is "equally_weighted_portfolio".
        """

        super().__init__(name=name)

    def fit(self, returns_assets: pd.DataFrame) -> "EquallyWeightedPortfolio":
        """
        Fits the portfolio by setting equal weights to each asset.

        This method assigns an equal weight to all assets provided in the
        `returns_assets` DataFrame. The resulting weights are normalized to sum
        to 1.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            A DataFrame where rows represent time periods and columns represent
            asset returns.

        Returns:
        --------
        EquallyWeightedPortfolio
            Returns the instance of the `EquallyWeightedPortfolio` class.

        Raises:
        -------
        ValueError
            If no assets are provided for weight allocation.
        """

        # Store essential details from the asset returns
        self.store_returns_var(returns_assets)

        num_assets = len(self.asset_list)
        if num_assets == 0:
            msg = "No assets to allocate weights."
            logger.error(msg)
            raise ValueError(msg)

        # Calculate equal weights for assets
        self._w = np.ones(num_assets) / num_assets

        # Store the asset weights as a dictionary
        self.asset_weights = dict(zip(self.asset_list, self._w))

        logger.debug(f"Assigned equal weights: {self.asset_weights}")

        return self

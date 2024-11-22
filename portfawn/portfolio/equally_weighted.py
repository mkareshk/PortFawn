import logging

import numpy as np
import pandas as pd

from .base import PortfolioBase

logger = logging.getLogger(__name__)


class EquallyWeightedPortfolio(PortfolioBase):

    def __init__(self, name: str = "Equally Weighted Portfolio") -> None:
        """
        Initializes the EquallyWeightedPortfolio with a given name.

        Args:
            name (str, optional): Name of the portfolio. Defaults to "equally_weighted_portfolio".
        """

        super().__init__(name=name)

    def fit(self, returns_assets: pd.DataFrame) -> "EquallyWeightedPortfolio":
        """
        Fits the portfolio by setting equal weights to each asset.

        This method assigns an equal weight to all assets provided in the
        `returns_assets` DataFrame. The resulting weights are normalized to sum
        to 1.

        Args:
            returns_assets (pd.DataFrame): A DataFrame where rows represent time periods
                and columns represent asset returns.

        Returns:
            EquallyWeightedPortfolio: The instance of the `EquallyWeightedPortfolio` class.

        Raises:
            ValueError: If no assets are provided for weight allocation.
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

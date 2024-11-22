import logging

import numpy as np
import pandas as pd

from .base import PortfolioBase

logger = logging.getLogger(__name__)


class RandomPortfolio(PortfolioBase):

    def __init__(self, name: str = "Randomly Weighted Portfolio") -> None:
        """
        Initializes the RandomPortfolio with a given name.

        Args:
            name (str, optional): Name of the portfolio. Defaults to "random_portfolio".
        """

        super().__init__(name)

    def fit(self, returns_assets: pd.DataFrame) -> "RandomPortfolio":
        """
        Fits the portfolio by assigning random weights to each asset.

        The method calculates random weights for the assets provided in the
        `returns_assets` DataFrame. The weights are normalized to ensure they
        sum to 1. The asset weights are then stored in the `asset_weights`
        attribute as a dictionary.

        Args:
            returns_assets (pd.DataFrame): A DataFrame where rows represent time periods
                and columns represent asset returns.

        Returns:
            RandomPortfolio: The instance of the `RandomPortfolio` class.

        Raises:
            ValueError: If the asset list is empty or invalid weights are generated.
        """

        # Store essential details from the asset returns
        self.store_returns_var(returns_assets)

        if len(self.asset_list) == 0:
            msg = "Asset list is empty."
            logger.error(msg)
            raise ValueError(msg)

        # Generate random weights for assets
        w = np.random.uniform(low=0.0, high=1.0, size=len(self.asset_list))
        sum_w = np.sum(w)
        if sum_w == 0 or np.any(np.isnan(w)):
            msg = "Invalid weights calculated."
            logger.error(msg)
            raise ValueError(msg)
        self._w = w / sum_w

        # Validate weights
        if not np.isclose(np.sum(self._w), 1):
            msg = "Random weights do not sum to 1 after normalization."
            logger.error(msg)
            raise ValueError(msg)

        # Store the asset weights as a dictionary
        self.asset_weights = dict(zip(self.asset_list, self._w))

        logger.debug(f"Assigned random weights: {self.asset_weights}")

        return self

import logging

import numpy as np
import pandas as pd

from .base import PortfolioBase

logger = logging.getLogger(__name__)


class RandomPortfolio(PortfolioBase):
    """
    A portfolio class that assigns random weights to assets.

    This class inherits from `PortfolioBase` and is used for creating a portfolio
    with randomly generated weights for a given set of assets. It ensures that
    the weights are normalized to sum up to 1.

    Attributes:
    -----------
    asset_list : list
        List of asset names available in the portfolio.
    asset_weights : dict
        Dictionary of assets and their corresponding weights.
    _w : np.ndarray
        Array of weights assigned to assets.
    """

    def __init__(self, name: str = "random_portfolio") -> None:
        """
        Initializes the RandomPortfolio with a given name.

        Parameters:
        -----------
        name : str, optional
            Name of the portfolio. Default is "random_portfolio".
        """

        super().__init__(name)

    def fit(self, returns_assets: pd.DataFrame) -> "RandomPortfolio":
        """
        Fits the portfolio by assigning random weights to each asset.

        The method calculates random weights for the assets provided in the
        `returns_assets` DataFrame. The weights are normalized to ensure they
        sum to 1. The asset weights are then stored in the `asset_weights`
        attribute as a dictionary.

        Parameters:
        -----------
        returns_assets : pd.DataFrame
            A DataFrame where rows represent time periods and columns represent
            asset returns.

        Returns:
        --------
        RandomPortfolio
            Returns the instance of the `RandomPortfolio` class.

        Raises:
        -------
        ValueError
            If the asset list is empty or invalid weights are generated.
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

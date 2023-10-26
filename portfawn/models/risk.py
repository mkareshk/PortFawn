from typing import Tuple

import numpy as np
import pandas as pd


class RiskModel:
    """
    A class representing a risk assessment model based on sampling techniques.

    Attributes:
    -----------
    sampling_type : str, default="standard"
        Type of sampling to be used. Possible values include "standard".
    sample_num : int, default=1000
        Number of samples to be drawn.
    sample_size : int, default=10
        Size of each sample.
    agg_func : str, default="median"
        Aggregation function to be applied on the samples. E.g., "median".

    Example:
    --------
    >>> model = RiskModel()
    >>> model.sampling_type
    'standard'
    >>> model.sample_num
    1000
    >>> model.sample_size
    10
    >>> model.agg_func
    'median'
    """

    def __init__(self,
                 sampling_type: str = "standard",
                 sample_num: int = 1000,
                 sample_size: int = 10,
                 agg_func: str = "median") -> None:
        """
        Initializes the RiskModel with the specified parameters.

        Parameters:
        -----------
        sampling_type : str, default="standard"
            Type of sampling to be used.
        sample_num : int, default=1000
            Number of samples to be drawn.
        sample_size : int, default=10
            Size of each sample.
        agg_func : str, default="median"
            Aggregation function to be applied on the samples.
        """
        self.sampling_type = sampling_type
        self.sample_num = sample_num
        self.sample_size = sample_size
        self.agg_func = agg_func

    def evaluate(self, returns):

        if self.sampling_type == "standard":  # simple, but unstable
            return self.standard(returns=returns)

        elif self.sampling_type == "bootstrapping":  # for robust stats
            return self.bootstrapping(returns=returns)

        else:
            raise NotImplementedError

    def standard(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Computes the mean (linear biases) and covariance (quadratic biases) of the provided returns.

        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing:
            1. pd.Series: Mean returns (linear biases) of the assets.
            2. pd.DataFrame: Covariance matrix (quadratic biases) of the assets.

        Example:
        --------
        >>> # Using a mock DataFrame for demonstration
        ... returns = pd.DataFrame({
        ...     'A': [0.01, 0.02, -0.01],
        ...     'B': [0.02, -0.01, 0.03]
        ... })
        ... linear_biases, quadratic_biases = standard(None, returns)
        >>> linear_biases
        A    0.006667
        B    0.013333
        dtype: float64
        >>> quadratic_biases
                A         B
        A  0.000178 -0.000178
        B -0.000178  0.000378
        """

        # Calculate the mean of returns for each asset
        linear_biases = returns.mean()

        # Calculate the covariance matrix of returns
        quadratic_biases = returns.cov()

        return linear_biases, quadratic_biases

    def bootstrapping(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Perform bootstrapping on returns to compute linear and quadratic biases.

        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing:
            1. pd.Series: Linear biases of the assets based on bootstrapped samples.
            2. pd.DataFrame: Quadratic biases (covariance) of the assets based on bootstrapped samples.

        Example:
        --------
        >>> # Using a mock DataFrame for demonstration
        ... returns = pd.DataFrame({
        ...     'A': [0.01, 0.02, -0.01, 0.015, 0.03],
        ...     'B': [0.02, -0.01, 0.03, 0.01, 0.025]
        ... })
        ... mock_instance = type('', (), {})()  # create a mock instance
        ... mock_instance.sample_num = 3
        ... mock_instance.sample_size = 2
        ... mock_instance.agg_func = 'median'
        ... linear_biases, quadratic_biases = bootstrapping(mock_instance, returns)
        >>> linear_biases
        A    0.0175
        B    0.0175
        dtype: float64
        """

        # List to store means and covariances for each sample
        linear_list = []
        quadratic_list = []

        # Extract samples and compute statistics
        for _ in range(self.sample_num):
            sample = returns.sample(n=self.sample_size, replace=True)
            agg_function = getattr(sample, self.agg_func)

            linear_list.append(agg_function())
            quadratic_list.append(sample.cov())

        # Compute aggregated statistics across all samples
        return_df = pd.DataFrame(linear_list)
        linear_biases = getattr(return_df, self.agg_func)()

        risk_matrix = np.array([cov_matrix.to_numpy()
                               for cov_matrix in quadratic_list])
        risk_aggregator = getattr(np, self.agg_func)

        risk_matrix = risk_aggregator(risk_matrix, axis=0)
        quadratic_biases = pd.DataFrame(
            risk_matrix, index=return_df.columns, columns=return_df.columns)

        return linear_biases, quadratic_biases

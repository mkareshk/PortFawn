from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class MeanVarianceRiskModel:
    """
    A risk assessment model based on sampling techniques for the mean-variance model.

    Attributes:
    -----------
    sampling_type : str, default="standard"
        Type of sampling to be used. Possible values include "standard" and "bootstrapping".
    sample_num : int, default=1000
        Number of samples to be drawn.
    sample_size : int, default=100
        Size of each sample.
    agg_func : str, default="mean"
        Aggregation function to be applied on the samples. Default is "mean".
    random_state : int or None, default=None
        Random state for reproducibility.
    """

    def __init__(
        self,
        sampling_type: str = "standard",
        sample_num: int = 1000,
        sample_size: int = 100,
        agg_func: str = "mean",
        random_state: int = None,
    ) -> None:
        """
        Initialize the MeanVarianceRiskModel with the specified parameters.

        Parameters:
        -----------
        sampling_type : str, default="standard"
            Type of sampling to be used. Possible values include "standard" and "bootstrapping".
        sample_num : int, default=1000
            Number of samples to be drawn.
        sample_size : int, default=100
            Size of each sample.
        agg_func : str, default="mean"
            Aggregation function to be applied on the samples. Must be "mean".
        random_state : int or None, default=None
            Random state for reproducibility.
        """
        VALID_SAMPLING_TYPES = {"standard", "bootstrapping"}
        if sampling_type not in VALID_SAMPLING_TYPES:
            raise ValueError(
                f"Invalid sampling_type '{sampling_type}'. Must be one of {VALID_SAMPLING_TYPES}."
            )

        VALID_AGG_FUNCS = {"mean"}
        if agg_func not in VALID_AGG_FUNCS:
            raise ValueError(
                f"Invalid agg_func '{agg_func}'. Must be one of {VALID_AGG_FUNCS}."
            )

        self.sampling_type = sampling_type
        self.sample_num = sample_num
        self.sample_size = sample_size
        self.agg_func = agg_func
        self.random_state = random_state

    def evaluate(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Evaluate the risk model using the provided returns data.

        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing:
            1. pd.Series: Expected returns of the assets.
            2. pd.DataFrame: Covariance matrix of the assets.
        """
        if self.sampling_type == "standard":
            return self._standard(returns=returns)
        elif self.sampling_type == "bootstrapping":
            return self._bootstrapping(returns=returns)
        else:
            raise NotImplementedError(
                f"Sampling type '{self.sampling_type}' is not implemented."
            )

    def _standard(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute the expected returns and covariance matrix of the provided returns.

        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing:
            1. pd.Series: Expected returns of the assets.
            2. pd.DataFrame: Covariance matrix of the assets.
        """
        returns = returns.dropna()
        expected_returns = returns.mean()
        covariance_matrix = returns.cov()
        return expected_returns, covariance_matrix

    def _bootstrapping(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Perform bootstrapping on returns to compute expected returns and covariance matrix.

        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing the returns of assets.

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            A tuple containing:
            1. pd.Series: Expected returns of the assets.
            2. pd.DataFrame: Covariance matrix of the assets.
        """
        returns = returns.dropna()

        if self.sample_size <= returns.shape[1]:
            raise ValueError("Sample size must be greater than the number of assets.")

        def compute_sample(i):
            rng = (
                np.random.default_rng(self.random_state + i)
                if self.random_state is not None
                else np.random.default_rng()
            )
            sample = returns.sample(
                n=self.sample_size, replace=True, random_state=rng.integers(1e9)
            )
            expected_returns = sample.mean()
            covariance_matrix = sample.cov()
            return expected_returns, covariance_matrix

        results = Parallel(n_jobs=-1)(
            delayed(compute_sample)(i) for i in range(self.sample_num)
        )

        expected_returns_list, covariance_matrices_list = zip(*results)
        expected_returns_df = pd.DataFrame(expected_returns_list)
        covariance_matrices = np.stack(
            [cov.values for cov in covariance_matrices_list], axis=0
        )

        expected_returns = expected_returns_df.mean()

        covariance_matrix_mean = np.mean(covariance_matrices, axis=0)
        covariance_matrix_psd = self._nearest_positive_semidefinite(
            covariance_matrix_mean
        )
        covariance_matrix = pd.DataFrame(
            covariance_matrix_psd, index=returns.columns, columns=returns.columns
        )

        return expected_returns, covariance_matrix

    @staticmethod
    def _nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive semi-definite matrix to the input.

        Parameters:
        -----------
        matrix : np.ndarray
            Symmetric matrix.

        Returns:
        --------
        np.ndarray
            Nearest positive semi-definite matrix.
        """
        sym_matrix = (matrix + matrix.T) / 2
        eigvals, eigvecs = np.linalg.eigh(sym_matrix)
        eigvals[eigvals < 0] = 0
        psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return psd_matrix

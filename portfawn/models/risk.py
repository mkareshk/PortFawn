import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskModel:
    def __init__(
        self,
        type: str = "standard",
        sample_num: int = 100,
        sample_size: int = 20,
        agg_func: str = "median",
    ) -> None:

        self._type = type
        self._sample_num = sample_num
        self._sample_size = sample_size
        self._agg_func = agg_func

    def evaluate(self, returns_data):

        returns = returns_data.returns

        if self._type == "standard":  # simple, but unstable
            return self.standard(returns=returns)

        elif self._type == "bootstrapping":  # for robust stats
            return self.bootstrapping(returns=returns)

        else:
            raise NotImplementedError

    def standard(self, returns):

        expected_return = returns.mean()
        expected_cov = returns.cov()

        return expected_return, expected_cov

    def bootstrapping(self, returns):

        return_list = []
        risk_list = []

        for _ in range(self._sample_num):

            sample = returns.sample(n=self._sample_size)

            return_list.append(eval(f"sample.{self._agg_func}()"))
            risk_list.append(sample.cov())

        return_df = pd.DataFrame(return_list)
        expected_return = eval(f"return_df.{self._agg_func}()")

        risk_matrix = np.array([i.to_numpy() for i in risk_list])
        risk_matrix = eval(f"np.{self._agg_func}(risk_matrix, axis=0)")
        expected_cov = pd.DataFrame(risk_matrix)
        expected_cov.columns = return_df.columns
        expected_cov.index = return_df.columns

        return expected_return, expected_cov

    @property
    def type(self):
        return self._type

    @property
    def sample_num(self):
        return self._sample_num

    @property
    def sample_size(self):
        return self._sample_size

    @property
    def agg_func(self):
        return self._agg_func

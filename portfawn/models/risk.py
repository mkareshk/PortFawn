import numpy as np
import pandas as pd


class RiskModel:
    def __init__(
        self,
        type: str = "standard",
        sample_num: int = 100,
        sample_size: int = 20,
        agg_func: str = "median",
    ) -> None:
        self.type = type
        self._sample_num = sample_num
        self._sample_size = sample_size
        self._agg_func = agg_func

    def sample(self, returns):

        if self.type == "standard":  # simple, but unstable
            self.standard(returns=returns)

        elif self.type == "bootstrapping":  # for robust stats
            self.bootstrapping(returns=returns)

        else:
            raise NotImplementedError

    @property
    def expected_returns(self):
        return self._expected_return

    @property
    def expected_risk(self):
        return self._expected_risk

    def standard(self, returns):
        self._expected_return = returns.mean()
        self._expected_risk = returns.cov()

    def bootstrapping(self, returns):
        return_list = []
        risk_list = []
        for _ in range(self._sample_num):

            sample = returns.sample(n=self._sample_size)

            return_list.append(eval(f"sample.{self._agg_func}()"))
            risk_list.append(sample.cov())

        return_df = pd.DataFrame(return_list)
        self._expected_return = eval(f"return_df.{self._agg_func}()")

        risk_matrix = np.array([i.to_numpy() for i in risk_list])
        self._expected_risk = pd.DataFrame(risk_matrix)
        self._expected_risk.columns = return_df.columns
        self._expected_risk.index = return_df.columns

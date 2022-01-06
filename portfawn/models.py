import numpy as np
import pandas as pd
from dwave.system import DWaveCliqueSampler, DWaveSampler, neal


class SamplingParams:
    def __init__(self) -> None:
        pass


class ClassicOptParams:
    def __init__(
        self,
        max_iter: int = 1000,
        disp: bool = False,
        ftol: float = 1e-10,
        target_return: float = 0.1,
        target_risk: float = 0.1,
        weight_bound: tuple = (0.0, 1.0),
    ) -> None:

        self._type = "classic"

        self._max_iter = max_iter
        self._disp = disp
        self._ftol = ftol
        self._target_return = target_return
        self._target_risk = target_risk
        self._weight_bound = weight_bound

    @property
    def params(self):
        return {
            "maxiter": self._max_iter,
            "disp": self._disp,
            "ftol": self._ftol,
            "target_return": self._target_return,
            "target_risk": self._target_risk,
            "weight_bound": self._weight_bound,
        }

    @property
    def type(self):
        self._type


class QuantumOptModel:
    def __init__(self, backend: str = "neal", annealing_time: int = 100) -> None:

        self._type = "quantum"

        self._backend = backend
        self._annealing_time = annealing_time

        if self._backend == "neal":
            self._sampler = neal.SimulatedAnnealingSampler()
        elif self._backend == "qpu":
            self._sampler = DWaveCliqueSampler()
        else:
            raise NotImplementedError

    def optimize(self, risk_model):

        # risk
        risk_term = np.triu(risk_model.expected_risk, k=1)

        # returns
        returns_term = np.zeros(risk_model.expected_risk.shape, float)
        np.fill_diagonal(returns_term, -risk_model.expected_returns)

        # Q
        Q = risk_term + returns_term

        # Sampling
        samples = self._sampler.sample_qubo(Q)

        w = np.array(list(samples.first.sample.values())).reshape(self.weight_shape)
        if not sum(w):
            w = np.ones(self.weight_shape)

        self._asset_weights = w / np.sum(w)

    @property
    def params(self):
        return {"annealing_time": self._annealing_time}

    @property
    def type(self):
        self._type

    @property
    def asset_weights(self):
        self._asset_weights


class EconomicModel:
    def __init__(self, risk_free_rate: float = 0.0, annualized_days: int = 252) -> None:
        self._risk_free_rate = risk_free_rate
        self._annualized_days = annualized_days

    @property
    def risk_free_rate(self):
        return self._risk_free_rate

    @property
    def annualized_days(self):
        return self._annualized_days


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

import logging

import neal
import numpy as np
import scipy.optimize as sco
from dwave.system import DWaveCliqueSampler

from portfawn.models.economic import EconomicModel

logger = logging.getLogger(__name__)


class QuantumOptModel:
    def __init__(
        self, objective, backend: str = "neal", annealing_time: int = 100
    ) -> None:

        self._type = "quantum"

        self._objective = objective
        self._backend = backend
        self._annealing_time = annealing_time

        if self._objective not in ["BMOP"]:
            raise NotImplementedError

        if self._backend == "neal":
            self._sampler = neal.SimulatedAnnealingSampler()

        elif self._backend == "qpu":
            self._sampler = DWaveCliqueSampler()

        else:
            raise NotImplementedError

    def optimize(self, expected_return, expected_cov):

        weight_shape = (len(expected_return), 1)

        asset_cov = expected_cov.to_numpy()
        asset_returns = expected_return.to_numpy()

        # risk
        risk_term = np.triu(asset_cov, k=1)

        # returns
        returns_term = np.zeros(asset_cov.shape, float)
        np.fill_diagonal(returns_term, -asset_returns)

        # Q
        Q = risk_term + returns_term

        # Sampling
        samples = self._sampler.sample_qubo(Q)

        w = np.array(list(samples.first.sample.values())).reshape(weight_shape)
        if not sum(w):
            w = np.ones(weight_shape)

        return w / np.sum(w)

    @property
    def type(self):
        self._type

    @property
    def objective(self):
        self._objective

    @property
    def params(self):
        return {"annealing_time": self._annealing_time}

    @property
    def sampler(self):
        self._sampler


class ClassicOptModel:
    def __init__(
        self,
        objective,
        economic_model=EconomicModel(),
        scipy_params: dict = {"maxiter": 1000, "disp": False, "ftol": 1e-10},
        target_return: float = 0.1,
        target_sd: float = 0.1,
        weight_bound: tuple = (0.0, 1.0),
        init_point=None,
    ) -> None:

        self._type = "classic"

        self._objective = objective
        self._economic_model = economic_model
        self._scipy_params = scipy_params
        self._target_return = target_return
        self._target_sd = target_sd
        self._weight_bound = weight_bound
        self._init_point = init_point

        if self._objective not in ["EWP", "MRP", "MVP", "MSRP"]:
            raise NotImplementedError

    def optimize(self, expected_return, expected_cov):

        if self._objective == "EWP":
            weight_shape = (len(expected_return), 1)
            return self._ewp(weight_shape)

        elif self._objective in ["MRP", "MVP", "MSRP"]:
            asset_cov = expected_cov.to_numpy()
            asset_returns = expected_return.to_numpy()
            return self._opt_portfolio(asset_returns, asset_cov)

    def _ewp(self, weight_shape):
        w = np.ones(weight_shape)
        return w / np.sum(w)

    def _opt_portfolio(self, asset_returns, asset_cov):

        asset_num = asset_returns.shape[0]

        constraints = []

        constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})  # sum(w) = 1

        if self._objective == "MRP":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: w.T.dot(asset_returns) - self._target_return,
                }
            )  # reach a target return
            cost_function = lambda w: -asset_returns.dot(w)

        elif self._objective == "MVP":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: self._target_sd - asset_returns.dot(w),
                }
            )  # reach a target risk
            cost_function = lambda w: np.sqrt(w.T.dot(asset_cov).dot(w))

        elif self._objective == "MSRP":  # no additional constraint
            cost_function = lambda w: -(
                asset_returns.dot(w) - self._economic_model.risk_free_rate
            ) / np.sqrt(w.T.dot(asset_cov).dot(w))

        else:
            raise NotImplementedError

        # optimization bounds - use the same bounds for all assets
        weight_bounds = tuple(self._weight_bound for _ in range(asset_num))

        # init point
        if not self._init_point:
            init_point = np.random.random(size=asset_num)

        # optimization
        result = sco.minimize(
            cost_function,
            init_point,
            method="SLSQP",
            bounds=weight_bounds,
            constraints=constraints,
            options=self._scipy_params,
        )

        w = result["x"].reshape(asset_num, 1)
        return w / np.sum(w)

    @property
    def scipy_params(self):
        return self._scipy_params

    @property
    def type(self):
        self._type

    @property
    def objective(self):
        self._objective

    @property
    def economic_model(self):
        self._economic_model

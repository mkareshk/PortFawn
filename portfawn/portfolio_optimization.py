import time

import neal
import dimod
import numpy as np
import scipy.optimize as sco
from sklearn.preprocessing import MinMaxScaler

# from dwave.system import DWaveSampler, EmbeddingComposite


class PortfolioOptimization:
    def __init__(
        self,
        portfolio_type,
        expected_return,
        expected_risk,
        risk_free_rate,
        optimization_params,
    ):

        self.portfolio_type = portfolio_type
        self.expected_return = expected_return.to_numpy()
        self.expected_risk = expected_risk.to_numpy()
        self.risk_free_rate = risk_free_rate
        self.optimization_params = optimization_params

        self.asset_num = self.expected_return.shape[0]
        self.weight_shape = (len(self.expected_return), 1)

    def optimize(self):

        if self.portfolio_type == "EWP":
            w = np.ones(self.weight_shape)

        elif self.portfolio_type in ["MRP", "MVP", "MSRP"]:
            w = self.real_value_weight()

        elif self.portfolio_type in ["SA"]:
            w = self.binary_value_weight()

        return self.normalized(w)  # sum(w) = 1, invest all capital

    def normalized(self, w):
        return w / np.sum(w)

    def real_value_weight(self):

        # constraints

        # sum(w) = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # reach a target return
        if self.portfolio_type == "MRP":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: w.T.dot(self.expected_risk).dot(w)
                    - self.optimization_params["target_return"],
                }
            )

        # reach a target risk
        elif self.portfolio_type == "MVP":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: self.optimization_params["target_risk"]
                    - self.expected_return.dot(w),
                }
            )

        elif self.portfolio_type == "MSRP":  # no additional constraint
            pass

        # optimization_type function
        if self.portfolio_type == "MRP":
            cost_function = self.cost_returns
        elif self.portfolio_type == "MVP":
            cost_function = self.cost_std
        elif self.portfolio_type == "MSRP":
            cost_function = self.cost_sharpe_ratio

        # optimization
        result = sco.minimize(
            cost_function,
            np.random.random(size=self.asset_num),  # random initial point
            method="SLSQP",
            bounds=tuple(
                self.optimization_params["weight_bound"] for _ in range(self.asset_num)
            ),  # use the same bound for all assets
            constraints=constraints,
            options=self.optimization_params["scipy_params"],
        )

        return result["x"].reshape(self.asset_num, 1)

    def cost_sharpe_ratio(self, weights):
        return -(self.expected_return.dot(weights) - self.risk_free_rate) / np.sqrt(
            weights.T.dot(self.expected_risk).dot(
                weights
            )  # add '-' since we aim to minimize
        )

    def cost_returns(self, weights):
        # add '-' since we aim to minimize
        return -self.expected_return.dot(weights)

    def cost_std(self, weights):
        return np.sqrt(weights.T.dot(self.expected_risk).dot(weights))

    def binary_value_weight(self):

        # risk
        risk_term = np.triu(self.expected_risk, k=1)

        # returns
        returns_term = np.zeros(self.expected_risk.shape, float)
        np.fill_diagonal(returns_term, -self.expected_return)

        # Q
        Q = risk_term + returns_term

        # Sampling
        sampler = neal.SimulatedAnnealingSampler()
        samples = sampler.sample_qubo(Q)

        w = np.array(list(samples.first.sample.values())).reshape(self.weight_shape)
        if not sum(w):
            w = np.ones(self.weight_shape)

        return w

import time

import neal
import dimod
import numpy as np
import scipy.optimize as sco

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

        if self.portfolio_type == "Equal":
            w = np.ones(self.weight_shape)

        elif self.portfolio_type in ["MR", "MV", "MSR"]:
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
        if self.portfolio_type == "MR":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: w.T.dot(self.expected_risk).dot(w) - self.optimization_params["target_return"],
                }
            )

        # reach a target risk
        elif self.portfolio_type == "MV":
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: self.optimization_params["target_risk"] - self.expected_return.dot(w),
                }
            )

        elif self.portfolio_type == "MSR":  # no additional constraint
            pass

        # optimization_type function
        if self.portfolio_type == "MR":
            cost_function = self.cost_returns
        elif self.portfolio_type == "MV":
            cost_function = self.cost_std
        elif self.portfolio_type == "MSR":
            cost_function = self.cost_sharpe_ratio

        # optimization
        result = sco.minimize(
            cost_function,
            np.random.random(size=self.asset_num),  # random initial point
            method="SLSQP",
            bounds=tuple(
                self.optimization_params["weight_bound"] for _ in range(self.asset_num)),  # use the same bound for all assets
            constraints=constraints,
            options=self.optimization_params["scipy_params"],
        )

        return result["x"].reshape(self.asset_num, 1)

    def cost_sharpe_ratio(self, weights):
        return -(self.expected_return.dot(weights) - self.risk_free_rate) / np.sqrt(
            weights.T.dot(self.expected_risk).dot(
                weights)  # add '-' since we aim to minimize
        )

    def cost_returns(self, weights):
        # add '-' since we aim to minimize
        return -self.expected_return.dot(weights)

    def cost_std(self, weights):
        return np.sqrt(weights.T.dot(self.expected_risk).dot(weights))

    def binary_value_weight(self):

        risk_term = np.triu(self.expected_risk, k=1)
        risk_max = np.max(risk_term)
        risk_term = risk_term / risk_max
        returns_term = np.zeros(self.expected_risk.shape, float)
        np.fill_diagonal(returns_term, self.expected_return)
        returns_term = returns_term / risk_max
        Q = risk_term + returns_term

        sampler = neal.SimulatedAnnealingSampler()
        samples = sampler.sample_qubo(Q)
        print(samples)
        return np.array(list(samples.first.sample.values())).reshape(-1, 1)

        # def optimized_binary(self, optimization_type):

        #     num_read = 500

        #     # problem definition
        #     Q = {}
        #     h = {}
        #     for i in range(len(self.returns_mean)):
        #         Q.update({(i, i): self.returns_mean[i]})

        #     J = {}
        #     for i in range(len(self.returns_mean)):
        #         for j in range(i + 1, len(self.returns_mean)):
        #             Q.update({(i, j): self.returns_cov[i][j]})

        #     problem = dimod.BinaryQuadraticModel(h, J, dimod.Vartype.SPIN)

        #     # sampling
        #     if optimization_type == "binary_sa":
        #         sampler = neal.SimulatedAnnealingSampler()
        #     elif optimization_type == "binary_qpu":
        #         sampler = EmbeddingComposite(DWaveSampler(qpu=True))

        #     sampleset = sampler.sample_qubo(Q, num_reads=num_read)
        #     return np.array([i for i in sampleset.first.sample.values()])

        # def optimized_real(self, optimization_type, cost_param):
        #     args = (self.returns_mean, self.returns_cov)
        #     weight_bound = (0.0, 1.0)
        #     weight_bounds = tuple(weight_bound for asset in range(self.asset_num))
        #     initial_point = np.random.random(size=self.asset_num)

        #     # constraints
        #     constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # \sig{w_i} = 1
        #     if optimization_type == "max_return":
        #         constraints.append(
        #             {
        #                 "type": "ineq",
        #                 "fun": lambda w: w.T.dot(self.returns_cov).dot(w) - cost_param,
        #             }
        #         )
        #     elif optimization_type == "min_variance":
        #         constraints.append(
        #             {"type": "ineq", "fun": lambda w: cost_param - self.returns_mean.dot(w)}
        #         )
        #     elif optimization_type == "max_sharpe_ratio":  # no additional constraint
        #         pass

        #     # optimization_type function
        #     if optimization_type == "max_return":
        #         cost_function = self.cost_returns
        #     elif optimization_type == "min_variance":
        #         cost_function = self.cost_std
        #     elif optimization_type == "max_sharpe_ratio":
        #         cost_function = self.cost_sharpe_ratio

        #     # optimization
        #     result = sco.minimize(
        #         cost_function,
        #         initial_point,
        #         args=args,
        #         method="SLSQP",
        #         bounds=weight_bounds,
        #         constraints=constraints,
        #         options=self.optimization_option,
        #     )

        #     return result

        # def random(self):
        #     if self.optimization_type in ["binary_qpu", "binary_sa"]:
        #         return np.random.choice([0, 1], size=(self.asset_num, 1))
        #     else:
        #         return np.random.uniform(low=0.0, high=1.0, size=(self.asset_num, 1))

        # def equal(self):
        #     np.random.seed(int(time.time()))
        #     return np.ones((self.asset_num, 1))

        # def optimize_iter(self, optimization_type, cost_param=0.1):
        #     if optimization_type == "random":
        #         w = self.random()
        #     elif optimization_type == "equal":
        #         w = self.equal()
        #     elif optimization_type in ["max_return", "min_variance", "max_sharpe_ratio"]:
        #         result = self.optimized_real(
        #             optimization_type=optimization_type, cost_param=cost_param
        #         )
        #         w = np.array(result["x"]).reshape(-1, 1)

        #     elif optimization_type in ["binary_qpu", "binary_sa"]:
        #         w = self.optimized_binary(optimization_type).reshape(-1, 1)
        #     return w / w.sum()

        # def optimize(self, optimization_type, cost_param=0.1, instance_num=20):

        #     if optimization_type == "min_variance":
        #         cost_param = 0.05
        #     elif optimization_type == "max_return":
        #         cost_param = 0.12

        #     w_list = []

        #     if optimization_type in ["binary_qpu", "binary_sa"]:
        #         instance_num = 1

        #     for i in range(instance_num):
        #         np.random.seed(int(np.sqrt(time.time()) * i))
        #         w_iter = self.optimize_iter(optimization_type, cost_param)
        #         w_list.append([w_iter])
        #     return np.array(w_list).mean(axis=1).mean(axis=0)

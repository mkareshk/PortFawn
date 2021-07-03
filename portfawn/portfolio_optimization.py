import time

# import neal
# import dimod
import numpy as np
import scipy.optimize as sco

# from dwave.system import DWaveSampler, EmbeddingComposite


class PortfolioOptimization:
    def __init__(self, optimization_method):
        self.optimization_method = optimization_method
        self.optimization_params = {"maxiter": 1000, "disp": False, "ftol": 1e-10}

    def optimize(self, expected_return, expected_risk, risk_free_rate):

        shape = (len(expected_return), 1)

        if self.optimization_method == "equal":
            return self.normalized(np.ones(shape))
        elif self.optimization_method == "random":

            return self.normalized(np.random.randint(low=0, high=100, size=shape))

    def normalized(self, w):
        return w / np.sum(w)

    # def calc_performance(self, weights, returns_mean, returns_cov):
    #     performance = {}
    #     performance.update({"return": returns_mean.dot(weights)})
    #     performance.update({"std": np.sqrt(weights.T.dot(returns_cov).dot(weights))})
    #     performance.update(
    #         {
    #             "sharpe_ratio": (performance["return"] - self.risk_free_rate)
    #             / performance["std"]
    #         }
    #     )

    #     return performance

    # def cost_sharpe_ratio(self, weights, returns_mean, returns_cov):
    #     return -self.calc_performance(weights, returns_mean, returns_cov)[
    #         "sharpe_ratio"
    #     ]

    # def cost_returns(self, weights, returns_mean, returns_cov):
    #     return -self.calc_performance(weights, returns_mean, returns_cov)["return"]

    # def cost_std(self, weights, returns_mean, returns_cov):
    #     return self.calc_performance(weights, returns_mean, returns_cov)["std"]

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

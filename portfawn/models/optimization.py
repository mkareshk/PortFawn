import dimod
import neal
import numpy as np
import scipy.optimize as sco
from dwave.system import DWaveCliqueSampler


class QuantumOptModel:
    """
    A class to perform optimization using quantum annealing techniques.

    Attributes:
        objective (str): The objective type for the optimization. Currently supports "BMOP".
        backend (str): The backend quantum system to use. Options are "neal" (simulated annealing) or "qpu" (D-Wave QPU).
        annealing_time (int): The annealing time for quantum annealing (in microseconds).
        num_reads (int): The number of reads (samples) from the sampler.
        num_sweeps (int): The number of sweeps for simulated annealing.
    """

    def __init__(
        self,
        objective,
        backend: str = "neal",
        annealing_time: int = 100,
        num_reads=1000,
        num_sweeps=10000,
    ) -> None:
        """
        Initialize the QuantumOptModel with the specified parameters.

        Args:
            objective (str): The optimization objective type.
            backend (str): Backend quantum system to use. Defaults to "neal".
            annealing_time (int): Annealing time in microseconds. Defaults to 100.
            num_reads (int): Number of reads to perform. Defaults to 1000.
            num_sweeps (int): Number of sweeps for simulated annealing. Defaults to 10000.

        Raises:
            NotImplementedError: If the objective or backend is unsupported.
        """

        self._validate_inputs(objective, backend)

        self._objective = objective
        self._backend = backend
        self._annealing_time = annealing_time
        self._num_reads = num_reads
        self._num_sweeps = num_sweeps

        if backend == "neal":
            self._sampler = neal.SimulatedAnnealingSampler()
        elif backend == "qpu":
            self._sampler = DWaveCliqueSampler()

    @staticmethod
    def _validate_inputs(objective, backend):
        """
        Validate the input parameters for the model.

        Args:
            objective (str): The optimization objective.
            backend (str): The backend to use.

        Raises:
            NotImplementedError: If the objective or backend is unsupported.
        """

        if objective not in ["BMOP"]:
            raise NotImplementedError(f"Objective '{objective}' not supported.")
        if backend not in ["neal", "qpu"]:
            raise NotImplementedError(f"Backend '{backend}' not supported.")

    def optimize(self, linear_biases: np.array, quadratic_biases: np.array) -> np.array:
        """
        Optimize the model using quantum annealing.

        Args:
            linear_biases (np.array): Array of linear biases (coefficients).
            quadratic_biases (np.array): 2D array of quadratic biases (coefficients).

        Returns:
            np.array: Optimized weights as a normalized array.

        Raises:
            ValueError: If no samples are returned by the sampler.
        """

        weight_shape = (len(linear_biases), 1)

        asset_cov = quadratic_biases.to_numpy()
        asset_returns = linear_biases.to_numpy()

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


class ClassicOptModel:
    """
    A class to perform classical optimization of portfolio weights.

    Attributes:
        objective (str): The optimization objective, such as "MRP" (mean return portfolio),
            "MVP" (minimum variance portfolio), or "MSRP" (maximum Sharpe ratio portfolio).
        risk_free_rate (float): The risk-free rate for optimization. Defaults to 0.0.
        scipy_params (dict): Parameters for the scipy optimizer, such as `maxiter`, `disp`, and `ftol`.
        target_return (float): Target return for the portfolio optimization. Defaults to 0.2.
        target_sd (float): Target standard deviation for the portfolio. Defaults to 0.2.
        weight_bound (tuple): Bounds for individual weights as (min, max). Defaults to (0.0, 1.0).
        init_point (np.array): Initial guess for the optimization. If None, a uniform allocation is used.
    """

    def __init__(
        self,
        objective,
        risk_free_rate=0.0,
        scipy_params: dict = {"maxiter": 1000, "disp": False, "ftol": 1e-10},
        target_return: float = 0.2,
        target_sd: float = 0.2,
        weight_bound: tuple = (0.0, 1.0),
        init_point=None,
    ) -> None:
        """
        Initialize the ClassicOptModel with specified parameters.

        Args:
            objective (str): The optimization objective. Must be one of "MRP", "MVP", or "MSRP".
            risk_free_rate (float): The risk-free rate for optimization. Defaults to 0.0.
            scipy_params (dict): Parameters for the scipy optimizer. Defaults to {"maxiter": 1000, "disp": False, "ftol": 1e-10}.
            target_return (float): The target portfolio return. Defaults to 0.2.
            target_sd (float): The target portfolio standard deviation. Defaults to 0.2.
            weight_bound (tuple): Bounds for individual weights as (min, max). Defaults to (0.0, 1.0).
            init_point (np.array): Initial guess for the optimizer. Defaults to None (uniform allocation).

        Raises:
            NotImplementedError: If the specified objective is unsupported.
        """

        if objective not in ["MRP", "MVP", "MSRP"]:
            raise NotImplementedError(f"Objective '{objective}' not supported.")

        self._objective = objective
        self._risk_free_rate = risk_free_rate
        self._scipy_params = scipy_params
        self._target_return = target_return
        self._target_sd = target_sd
        self._weight_bound = weight_bound
        self._init_point = init_point

    def optimize(self, linear_biases: np.array, quadratic_biases: np.array) -> np.array:
        """
        Perform optimization of portfolio weights.

        Args:
            linear_biases (np.array): Expected returns for the assets.
            quadratic_biases (np.array): Covariance matrix of asset returns.

        Returns:
            np.array: Optimized portfolio weights as a normalized array.

        Raises:
            ValueError: If the optimization fails or the sum of weights is zero.

        Notes:
            - For "MRP" (Mean Return Portfolio), the objective is to minimize variance while achieving the target risk.
            - For "MVP" (Minimum Variance Portfolio), the objective is to minimize variance while keeping return below the target.
            - For "MSRP" (Maximum Sharpe Ratio Portfolio), the objective is to maximize the Sharpe ratio.
        """

        asset_num = len(linear_biases)
        weight_bounds = tuple(self._weight_bound for _ in range(asset_num))

        # Constraint: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if self._objective == "MRP":
            # Constraint: portfolio return >= target_return
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: linear_biases.dot(w) - self._target_return,
                }
            )

            def cost_function(w):
                return w.T.dot(quadratic_biases).dot(w)

        elif self._objective == "MVP":
            # Constraint: portfolio risk <= target_sd
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: self._target_sd
                    - np.sqrt(w.T.dot(quadratic_biases).dot(w)),
                }
            )

            def cost_function(w):
                return -linear_biases.dot(w)

        elif self._objective == "MSRP":

            def cost_function(w):
                portfolio_return = linear_biases.dot(w)
                portfolio_variance = w.T.dot(quadratic_biases).dot(w)
                if portfolio_variance == 0:
                    return np.inf
                sharpe_ratio = (portfolio_return - self._risk_free_rate) / np.sqrt(
                    portfolio_variance
                )
                return -sharpe_ratio

        else:
            raise NotImplementedError

        init_point = (
            self._init_point
            if self._init_point is not None
            else np.full(asset_num, 1.0 / asset_num)
        )

        result = sco.minimize(
            cost_function,
            init_point,
            method="SLSQP",
            bounds=weight_bounds,
            constraints=constraints,
            options=self._scipy_params,
        )

        if result.success:
            w = result.x
            total_weight = np.sum(w)
            if total_weight > 0:
                w = w / total_weight
            else:
                raise ValueError("Sum of weights is zero; cannot normalize.")
            return w.reshape(asset_num, 1)
        else:
            raise ValueError(f"Optimization failed: {result.message}")


class OptimizationModel:
    """
    A unified interface for performing optimization using quantum or classical methods.

    This class serves as a wrapper for both `QuantumOptModel` and `ClassicOptModel`,
    enabling the selection of optimization methods based on the objective.

    Attributes:
        objective (str): The optimization objective. Supported objectives are:
            - "BMOP" for quantum-based optimization.
            - "MRP" (Mean Return Portfolio) for classical optimization.
            - "MVP" (Minimum Variance Portfolio) for classical optimization.
            - "MSRP" (Maximum Sharpe Ratio Portfolio) for classical optimization.
        optimization_params (dict): A dictionary of parameters for the optimization process.
            These parameters differ based on the selected backend or objective.
        risk_free_rate (float): The risk-free rate, used in objectives like MSRP for Sharpe ratio calculation.
    """

    def __init__(
        self,
        objective: str,
        optimization_params: dict = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        """
        Initialize the OptimizationModel with the specified objective and parameters.

        Args:
            objective (str): The optimization objective. Supported objectives are "BMOP", "MRP", "MVP", and "MSRP".
            optimization_params (dict, optional): A dictionary of optimization parameters. Defaults to None.
                If None, default parameters for both quantum and classical backends will be used.
                Example parameters include:
                - Quantum: {"backend": "neal", "annealing_time": 100, "num_reads": 1000, "num_sweeps": 10000}.
                - Classical: {"maxiter": 1000, "disp": False, "ftol": 1e-10, "weight_bound": (0.0, 1.0)}.
            risk_free_rate (float): The risk-free rate. Defaults to 0.0.

        Raises:
            NotImplementedError: If the specified objective is unsupported.
        """

        if optimization_params is None:
            optimization_params = {
                "maxiter": 1000,
                "disp": False,
                "ftol": 1e-10,
                "backend": "neal",
                "annealing_time": 100,
                "num_reads": 1000,
                "num_sweeps": 10000,
                "weight_bound": (0.0, 1.0),
                "target_return": 0.15,
                "target_sd": 0.1,
                "init_point": None,
            }
        self.objective = objective
        self.optimization_params = optimization_params
        self.risk_free_rate = risk_free_rate

        if self.objective == "BMOP":
            self.optimizer = QuantumOptModel(
                objective=self.objective,
                backend=self.optimization_params.get("backend", "neal"),
                annealing_time=self.optimization_params.get("annealing_time", 100),
                num_reads=self.optimization_params.get("num_reads", 1000),
                num_sweeps=self.optimization_params.get("num_sweeps", 10000),
            )
        elif self.objective in ["MRP", "MVP", "MSRP"]:
            scipy_params = {
                k: v
                for k, v in self.optimization_params.items()
                if k in ["maxiter", "disp", "ftol"]
            }
            self.optimizer = ClassicOptModel(
                objective=self.objective,
                risk_free_rate=self.risk_free_rate,
                scipy_params=scipy_params,
                target_return=self.optimization_params.get("target_return", 0.15),
                target_sd=self.optimization_params.get("target_sd", 0.1),
                weight_bound=self.optimization_params.get("weight_bound", (0.0, 1.0)),
                init_point=self.optimization_params.get("init_point", None),
            )
        else:
            raise NotImplementedError(f"Objective '{self.objective}' not supported.")

    def optimize(self, linear_biases: np.array, quadratic_biases: np.array) -> np.array:
        """
        Perform optimization using the selected backend and objective.

        Args:
            linear_biases (np.array): Array of linear biases (e.g., expected returns for classical objectives).
            quadratic_biases (np.array): 2D array of quadratic biases (e.g., covariance matrix for classical objectives).

        Returns:
            np.array: The optimized weights as a normalized array.

        Raises:
            ValueError: If the sum of optimized weights is zero or optimization fails.

        Notes:
            - Quantum optimization ("BMOP") uses annealing to solve QUBO problems.
            - Classical optimization ("MRP", "MVP", "MSRP") uses `scipy.optimize.minimize` with SLSQP.
            - Ensure the provided biases match the objective type and backend's requirements.
        """

        w = self.optimizer.optimize(linear_biases, quadratic_biases)
        total_weight = np.sum(w)
        if total_weight > 0:
            w = w / total_weight
        else:
            raise ValueError("Sum of weights is zero; cannot normalize.")
        return w

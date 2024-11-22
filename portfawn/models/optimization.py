import dimod
import neal
import numpy as np
import scipy.optimize as sco
from dwave.system import DWaveCliqueSampler

from .utils import (
    annual_to_daily_return,
    annual_to_daily_volatility,
    validate_and_update_target_return,
    validate_and_update_target_sd,
)


class QuantumOptModel:

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

        # Calculate the required QUBO matrix
        quad_term = np.triu(quadratic_biases, k=1)
        lin_term = np.zeros(quadratic_biases.shape, float)
        np.fill_diagonal(lin_term, -linear_biases)
        Q = quad_term + lin_term

        # Create BQM from QUBO and sample
        bqm = dimod.BQM.from_qubo(Q, 0)
        samples = self._sampler.sample(
            bqm, num_reads=self._num_reads, num_sweeps=self._num_sweeps
        )

        # Extract the weights from the sample and normalize
        weight_shape = (len(linear_biases), 1)
        w = np.array(list(samples.first.sample.values())).reshape(weight_shape)
        if not sum(w):
            w = np.ones(weight_shape)
        return w / np.sum(w)


class ClassicOptModel:
    def __init__(
        self,
        objective: str,
        risk_free_rate: float = 0.0,
        scipy_params: dict = None,
        target_return: float = 0.1,
        target_sd: float = 0.1,
        weight_bounds: list = None,
        init_point: np.array = None,
    ) -> None:
        """
        Initialize the ClassicOptModel with specified parameters.

        Args:
            objective (str): The optimization objective. Must be one of "MRP", "MVP", or "MSRP".
            risk_free_rate (float): The risk-free rate for optimization. Defaults to 0.0.
            scipy_params (dict): Parameters for the scipy optimizer. Defaults to {"maxiter": 1000, "disp": False, "ftol": 1e-8}.
            target_return (float): The annual target portfolio return. Defaults to 0.1.
            target_sd (float): The annual target portfolio standard deviation. Defaults to0.1.
            weight_bounds (list): List of tuples specifying (min, max) bounds for each asset. Defaults to (0.0, 1.0) for all assets.
            init_point (np.array): Initial guess for the optimizer. Defaults to None (uniform allocation).

        Raises:
            ValueError: If an unsupported objective is provided.
        """
        if objective not in ["MRP", "MVP", "MSRP"]:
            raise ValueError(f"Objective '{objective}' not supported.")

        self._objective = objective
        self._risk_free_rate = risk_free_rate
        self._scipy_params = scipy_params or {
            "maxiter": 1000,
            "disp": False,
            "ftol": 1e-8,
        }
        self._target_return = annual_to_daily_return(target_return)
        self._target_sd = annual_to_daily_volatility(target_sd)
        self._weight_bounds = weight_bounds or [(0.0, 1.0)]
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
            - "MRP" (Maximum Return Portfolio) maximizes return while keeping risk below a target.
            - "MVP" (Minimum Variance Portfolio) minimizes risk while achieving at least the target return.
            - "MSRP" (Maximum Sharpe Ratio Portfolio) maximizes the Sharpe ratio.
        """

        asset_num = len(linear_biases)

        # Validate and update target return and standard deviation
        target_sd = validate_and_update_target_sd(self._target_sd, quadratic_biases)
        target_return = validate_and_update_target_return(
            self._target_return, linear_biases
        )

        # If uniform bounds, apply to all assets
        if len(self._weight_bounds) == 1:
            weight_bounds = self._weight_bounds * asset_num
        elif len(self._weight_bounds) != asset_num:
            raise ValueError(
                f"Length of weight_bounds must match the number of assets ({(self._weight_bounds)} != {asset_num})."
            )

        # Constraint: Weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if self._objective == "MRP":
            # Add inequality constraint to ensure portfolio variance is below target risk
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: target_sd
                    - np.sqrt(w.T.dot(quadratic_biases).dot(w)),
                }
            )

            def cost_function(w):
                return -linear_biases.dot(w)  # Maximize return

        elif self._objective == "MVP":
            # Add inequality constraint to ensure portfolio return is at least the target return
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: w.T.dot(linear_biases) - target_return,
                }
            )

            def cost_function(w):
                return w.T.dot(quadratic_biases).dot(w)  # Minimize variance

        elif self._objective == "MSRP":

            def cost_function(w):
                portfolio_return = linear_biases.dot(w)
                portfolio_variance = w.T.dot(quadratic_biases).dot(w)
                if portfolio_variance <= 1e-8:  # Handle near-zero variance gracefully
                    return np.inf
                sharpe_ratio = (portfolio_return - self._risk_free_rate) / np.sqrt(
                    portfolio_variance
                )
                return -sharpe_ratio  # Maximize Sharpe ratio

        # Initialize weights
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
                return (w / total_weight).reshape(-1, 1)  # Normalize weights
            else:
                raise ValueError("Sum of weights is zero; cannot normalize.")
        else:
            raise ValueError(f"Optimization failed: {result.message}")

    def _generate_random_init_point(self, asset_num: int) -> np.array:
        """
        Generate a random initial point that satisfies the constraints.

        Args:
            asset_num (int): Number of assets.

        Returns:
            np.array: Randomly generated initial weights.
        """
        bounds = self._weight_bounds
        lower_bounds, upper_bounds = zip(*bounds)

        # Generate random weights within bounds
        random_weights = np.random.uniform(
            low=lower_bounds, high=upper_bounds, size=asset_num
        )

        # Normalize weights to sum to 1
        normalized_weights = random_weights / np.sum(random_weights)

        return normalized_weights


class OptimizationModel:

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
                - Classical: {"maxiter": 1000, "disp": False, "ftol": 1e-10, "weight_bounds": [(0.0, 1.0)]}.
            risk_free_rate (float): The risk-free rate. Defaults to 0.0.

        Raises:
            NotImplementedError: If the specified objective is unsupported.
        """

        default_optimization_params = {
            "maxiter": 1000,
            "disp": False,
            "ftol": 1e-8,
            "backend": "neal",
            "annealing_time": 100,
            "num_reads": 1000,
            "num_sweeps": 10000,
            "weight_bounds": [(0.0, 1.0)],
            "target_return": 0.08,
            "target_sd": 0.08,
            "init_point": None,
        }
        if not optimization_params:
            optimization_params = default_optimization_params
        else:
            for k, v in default_optimization_params.items():
                if k not in optimization_params:
                    optimization_params[k] = v

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
                target_return=self.optimization_params["target_return"],
                target_sd=self.optimization_params["target_sd"],
                weight_bounds=self.optimization_params["weight_bounds"],
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

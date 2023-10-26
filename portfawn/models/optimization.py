import neal
import dimod
import numpy as np
import scipy.optimize as sco
from dwave.system import DWaveCliqueSampler


class QuantumOptModel:
    """
    This class represents a model that utilizes quantum optimization techniques.

    Attributes:
        objective (str): The objective type for the optimization.
        backend (str): The backend quantum system to use. Options are "neal" and "qpu".
        annealing_time (int): The annealing time.
        num_reads (int): The number of reads for the sampler.
        num_sweeps (int): The number of sweeps for the sampler.

    Raises:
        NotImplementedError: If the objective or backend provided is not supported.

    Example:
        >>> model = QuantumOptModel("BMOP")
        >>> linear_biases = np.array([0.5, -0.5])
        >>> quadratic_biases = np.array([[0.0, -0.5], [-0.5, 0.0]])
        >>> result = model.optimize(linear_biases, quadratic_biases)
        >>> np.sum(result)  # Confirming that the sum of result is approximately 1
        1.0
    """

    def __init__(self, objective, backend: str = "neal", annealing_time: int = 100, num_reads=1000, num_sweeps=10000) -> None:
        # Validating the given objective and backend
        self._validate_inputs(objective, backend)

        self._objective = objective
        self._backend = backend
        self._annealing_time = annealing_time
        self._num_reads = num_reads
        self._num_sweeps = num_sweeps

        # Set the sampler based on the backend
        if backend == "neal":
            self._sampler = neal.SimulatedAnnealingSampler()
        elif backend == "qpu":
            self._sampler = DWaveCliqueSampler()

    @staticmethod
    def _validate_inputs(objective, backend):
        """Validate the objective and backend provided to the constructor."""
        if objective not in ["BMOP"]:
            raise NotImplementedError(
                f"Objective '{objective}' not supported.")
        if backend not in ["neal", "qpu"]:
            raise NotImplementedError(f"Backend '{backend}' not supported.")

    def optimize(self, linear_biases: np.array, quadratic_biases: np.array) -> np.array:
        """
        Optimize the model based on provided biases.

        Args:
            linear_biases (np.array): The linear biases for the optimization.
            quadratic_biases (np.array): The quadratic biases for the optimization.

        Returns:
            np.array: The optimized weights.
        """
        # Calculate the required QUBO matrix
        quad_term = np.triu(quadratic_biases, k=1)
        lin_term = np.zeros(quadratic_biases.shape, float)
        np.fill_diagonal(lin_term, -linear_biases)
        Q = quad_term + lin_term

        # Create BQM from QUBO and sample
        bqm = dimod.BQM.from_qubo(Q, 0)
        samples = self._sampler.sample(
            bqm, num_reads=self._num_reads, num_sweeps=self._num_sweeps)

        # Extract the weights from the sample and normalize
        weight_shape = (len(linear_biases), 1)
        w = np.array(list(samples.first.sample.values())).reshape(weight_shape)
        if not sum(w):
            w = np.ones(weight_shape)
        return w / np.sum(w)


class ClassicOptModel:
    """
    ClassicOptModel optimizes asset weights using classical optimization techniques.

    Attributes:
        objective (str): The optimization objective ("MRP", "MVP", or "MSRP").
        risk_free_rate (float): Risk-free rate (default is 0.0).
        scipy_params (dict): Parameters for scipy optimization.
        target_return (float): Target return rate.
        target_sd (float): Target standard deviation.
        weight_bound (tuple): Tuple containing the minimum and maximum bounds for weights.
        init_point (np.array): Initial point for the optimizer.

    Raises:
        NotImplementedError: If the objective provided is not supported.

    Example:
        >>> model = ClassicOptModel("MVP")
        >>> linear_biases = np.array([0.05, 0.08])
        >>> quadratic_biases = np.array([[0.1, 0.03], [0.03, 0.12]])
        >>> result = model.optimize(linear_biases, quadratic_biases)
        >>> np.isclose(np.sum(result), 1.0)  # Confirming that the sum of result is approximately 1
        True
    """

    def __init__(self, objective, risk_free_rate=0.0,
                 scipy_params: dict = {"maxiter": 1000,
                                       "disp": False, "ftol": 1e-10},
                 target_return: float = 0.1,
                 target_sd: float = 0.1,
                 weight_bound: tuple = (0.0, 1.0),
                 init_point=None) -> None:

        if objective not in ["MRP", "MVP", "MSRP"]:
            raise NotImplementedError(
                f"Objective '{objective}' not supported.")

        self._objective = objective
        self._risk_free_rate = risk_free_rate
        self._scipy_params = scipy_params
        self._target_return = target_return
        self._target_sd = target_sd
        self._weight_bound = weight_bound
        self._init_point = init_point

    def optimize(self, linear_biases: np.array, quadratic_biases: np.array) -> np.array:
        """
        Optimize the asset weights based on provided biases.

        Args:
            linear_biases (np.array): The linear biases for the optimization.
            quadratic_biases (np.array): The quadratic biases for the optimization.

        Returns:
            np.array: The optimized weights.
        """

        # General constraint to ensure weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if self._objective == "MRP":
            constraints.append({
                "type": "ineq",
                "fun": lambda w: w.T.dot(linear_biases) - self._target_return
            })
            def cost_function(w): return -linear_biases.dot(w)

        elif self._objective == "MVP":
            constraints.append({
                "type": "ineq",
                "fun": lambda w: self._target_sd - linear_biases.dot(w)
            })

            def cost_function(w): return np.sqrt(
                w.T.dot(quadratic_biases).dot(w))

        elif self._objective == "MSRP":
            def cost_function(w): return -(
                linear_biases.dot(w) - self._risk_free_rate) / np.sqrt(w.T.dot(quadratic_biases).dot(w))

        else:
            raise NotImplementedError

        asset_num = len(linear_biases)
        weight_bounds = tuple(self._weight_bound for _ in range(asset_num))

        init_point = self._init_point if self._init_point is not None else np.random.random(
            size=asset_num)

        result = sco.minimize(
            cost_function,
            init_point,
            method="SLSQP",
            bounds=weight_bounds,
            constraints=constraints,
            options=self._scipy_params
        )

        return result["x"].reshape(asset_num, 1) / np.sum(result["x"])


class OptimizationModel:
    """
    OptimizationModel class that provides a unified interface for optimization 
    using either quantum or classical methods based on the specified objective.

    Attributes:
        objective (str): The optimization objective.
        optimization_params (dict): Parameters for the optimization.
        risk_free_rate (float): The risk-free rate.

    Raises:
        NotImplementedError: If the objective provided is not supported.

    Example:
        >>> opt_model = OptimizationModel("MVP")
        >>> linear_biases = np.array([0.05, 0.08])
        >>> quadratic_biases = np.array([[0.1, 0.03], [0.03, 0.12]])
        >>> result = opt_model.optimize(linear_biases, quadratic_biases)
        >>> np.isclose(np.sum(result), 1.0)  # Confirming that the sum of result is approximately 1
        True
    """

    def __init__(self, objective: str,
                 optimization_params: dict = {
                     "maxiter": 1000,
                     "disp": False,
                     "ftol": 1e-10,
                     "backend": "neal",
                     "annealing_time": 100,
                     "num_reads": 1000,
                     "num_sweeps": 10000,
                     "weight_bound": (0.0, 1.0),
                     "target_return": 0.1,
                     "target_sd": 0.05,
                     "init_point": None
                 },
                 risk_free_rate: float = 0.0) -> None:

        self.objective = objective
        self.optimization_params = optimization_params
        self.risk_free_rate = risk_free_rate

        if self.objective == "BMOP":
            self.optimizer = QuantumOptModel(
                objective=self.objective,
                backend=self.optimization_params["backend"],
                annealing_time=self.optimization_params["annealing_time"],
                num_reads=self.optimization_params["num_reads"],
                num_sweeps=self.optimization_params["num_sweeps"]
            )
        elif self.objective in ["MRP", "MVP", "MSRP"]:
            scipy_params = {
                k: v for k, v in self.optimization_params.items() if k in ["maxiter", "disp", "ftol"]
            }
            self.optimizer = ClassicOptModel(
                objective=self.objective,
                risk_free_rate=self.risk_free_rate,
                scipy_params=scipy_params,
                target_return=self.optimization_params["target_return"],
                target_sd=self.optimization_params["target_sd"],
                weight_bound=self.optimization_params["weight_bound"],
                init_point=self.optimization_params["init_point"]
            )
        else:
            raise NotImplementedError(
                f"Objective '{self.objective}' not supported.")

    def optimize(self, linear_biases: np.array, quadratic_biases: np.array) -> np.array:
        """
        Optimize based on linear and quadratic biases.

        Args:
            linear_biases (np.array): Linear biases.
            quadratic_biases (np.array): Quadratic biases.

        Returns:
            np.array: Optimized weights.
        """

        w = self.optimizer.optimize(linear_biases, quadratic_biases)
        w[w < 0.0001] = 0.0
        return w

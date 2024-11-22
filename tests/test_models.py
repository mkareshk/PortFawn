import numpy as np
import pytest

from portfawn.models import ClassicOptModel, QuantumOptModel


@pytest.mark.parametrize(
    "objective, backend, expected_exception",
    [
        ("BMOP", "neal", None),
        ("InvalidObjective", "neal", NotImplementedError),
        ("BMOP", "unsupported_backend", NotImplementedError),
    ],
)
def test_quantum_model_initialization(objective, backend, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            QuantumOptModel(objective, backend)
    else:
        model = QuantumOptModel(objective, backend)
        assert model._objective == objective
        assert model._backend == backend


@pytest.mark.parametrize(
    "objective, scipy_params, expected_exception",
    [
        ("MRP", {"maxiter": 1000}, None),
        ("InvalidObjective", {"maxiter": 1000}, ValueError),
    ],
)
def test_classic_model_initialization(objective, scipy_params, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            ClassicOptModel(objective, scipy_params=scipy_params)
    else:
        model = ClassicOptModel(objective, scipy_params=scipy_params)
        assert model._objective == objective


@pytest.mark.parametrize(
    "linear_biases, quadratic_biases, expected_exception",
    [
        (np.array([0.1, 0.2]), np.array([[0.01, 0.02], [0.02, 0.03]]), None),
        (np.array([0.1, 0.2]), np.array([[0.01, 0.02]]), ValueError),
    ],
)
def test_classic_model_optimize(linear_biases, quadratic_biases, expected_exception):
    model = ClassicOptModel("MRP")
    if expected_exception:
        with pytest.raises(expected_exception):
            model.optimize(linear_biases, quadratic_biases)
    else:
        weights = model.optimize(linear_biases, quadratic_biases)
        assert weights.sum() == pytest.approx(1.0), "Weights should sum to 1."

import pytest
from dafin import Returns

from portfawn.models.risk import RiskModel
from portfawn.models.optimization import ClassicOptModel, QuantumOptModel

from tests.utils import ASSET_LIST


returns_data = Returns(asset_list=ASSET_LIST)
risk_model = RiskModel()
expected_returns, expected_cov = risk_model.evaluate(returns_data)

quantum_optimizers = [
    QuantumOptModel(objective=o, backend=b) for o in ["BMOP"] for b in ["neal"]
]

classic_optimizers = [
    ClassicOptModel(objective=o) for o in ["EWP", "MRP", "MVP", "MSRP"]
]

optimizers = quantum_optimizers + classic_optimizers
ids = [o.objective for o in optimizers]


@pytest.mark.parametrize("optimizer", optimizers, ids=ids)
def test_quantum_optimization_neal(optimizer):
    w = optimizer.optimize(expected_returns, expected_cov)

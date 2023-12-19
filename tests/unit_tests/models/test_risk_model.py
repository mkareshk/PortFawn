import pytest
import pandas as pd
from dafin import Returns

from portfawn.models.risk import RiskModel
from tests.utils import ASSET_LIST


returns_data = Returns(asset_list=ASSET_LIST)


def test_risk_model_change_parms():
    risk_model = RiskModel(
        type="standard", sample_num=1, sample_size=2, agg_func="median"
    )

    assert risk_model.type == "standard"
    assert risk_model.sample_num == 1
    assert risk_model.sample_size == 2
    assert risk_model.agg_func == "median"


@pytest.mark.parametrize(
    "evaluation_type", ["standard", "bootstrapping"], ids=["standard", "bootstrapping"]
)
def test_risk_model_standard(evaluation_type):
    risk_model = RiskModel(type=evaluation_type)
    expected_returns, expected_cov = risk_model.evaluate(returns_data=returns_data)

    assert risk_model.type == evaluation_type
    assert isinstance(expected_returns, pd.Series)
    assert isinstance(expected_cov, pd.DataFrame)
    assert expected_returns.shape == (3,)
    assert expected_cov.shape == (3, 3)

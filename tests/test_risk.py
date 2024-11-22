import dafin
import pandas as pd
import pytest

from portfawn.models import MeanVarianceRiskModel


@pytest.fixture
def sample_returns():
    assets_list = ["SPY", "GLD", "BND"]
    date_start = "2010-01-01"
    date_end = "2022-12-30"
    data_instance = dafin.ReturnsData(assets_list)
    return data_instance.get_returns(date_start=date_start, date_end=date_end)


@pytest.mark.parametrize(
    "sampling_type, agg_func, expected_exception",
    [
        ("standard", "mean", None),
        ("unsupported", "mean", ValueError),
        ("standard", "unsupported", ValueError),
        ("bootstrapping", "mean", None),
    ],
)
def test_risk_model_initialization(sampling_type, agg_func, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            MeanVarianceRiskModel(sampling_type=sampling_type, agg_func=agg_func)
    else:
        model = MeanVarianceRiskModel(sampling_type=sampling_type, agg_func=agg_func)
        assert (
            model.sampling_type == sampling_type
        ), "Sampling type should be set correctly."


@pytest.mark.parametrize("sampling_type", ["standard", "bootstrapping"])
def test_risk_model_evaluate(sampling_type, sample_returns):
    model = MeanVarianceRiskModel(sampling_type=sampling_type)
    expected_returns, covariance = model.evaluate(sample_returns)
    assert not expected_returns.empty, "Expected returns should not be empty."
    assert not covariance.empty, "Covariance matrix should not be empty."

from portfawn.models.economic import EconomicModel


def test_economic_model_default_params():
    economic_model = EconomicModel()
    assert economic_model.risk_free_rate == 0.0
    assert economic_model.annualized_days == 252


def test_economic_model_modified_params():
    economic_model = EconomicModel(risk_free_rate=1, annualized_days=2)
    assert economic_model.risk_free_rate == 1
    assert economic_model.annualized_days == 2

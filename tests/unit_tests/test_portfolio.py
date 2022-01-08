import pytest

from portfawn.portfolio import Portfolio

asset_list = ["SPY", "GLD"]

quantum_objectives = ["BMOP"]
classic_objectives = ["EWP", "MRP", "MVP", "MSRP"]
objectives = quantum_objectives  # + classic_objectives


@pytest.mark.parametrize("objective", objectives, ids=objectives)
def test_portfolio(objective):
    portfolio = Portfolio(name=objective, objective=objective)
    result = portfolio.run(asset_list=asset_list)
    import logging

    logging.error(result)

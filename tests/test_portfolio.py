import dafin
import pandas as pd
import pytest

from portfawn.portfolio import (
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    RandomPortfolio,
)


@pytest.fixture
def sample_returns():
    assets_list = ["SPY", "GLD", "BND"]
    date_start = "2010-01-01"
    date_end = "2022-12-30"
    data_instance = dafin.ReturnsData(assets_list)
    return data_instance.get_returns(date_start=date_start, date_end=date_end)


def test_equally_weighted_portfolio(sample_returns):
    portfolio = EquallyWeightedPortfolio()
    portfolio.fit(sample_returns)
    assert sum(portfolio.asset_weights.values()) == pytest.approx(
        1.0
    ), "Weights should sum to 1."


def test_random_portfolio(sample_returns):
    portfolio = RandomPortfolio()
    portfolio.fit(sample_returns)
    assert sum(portfolio.asset_weights.values()) == pytest.approx(
        1.0
    ), "Weights should sum to 1."


def test_mean_variance_portfolio(sample_returns):
    portfolio = MeanVariancePortfolio()
    portfolio.fit(sample_returns)
    assert sum(portfolio.asset_weights.values()) == pytest.approx(
        1.0
    ), "Weights should sum to 1."

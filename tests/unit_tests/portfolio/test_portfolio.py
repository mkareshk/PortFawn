import pytest

from portfawn import MeanVariancePortfolio, PlotPortfolio
from ..utils import ASSET_LIST, OBJECTIVES, check_figure


@pytest.mark.parametrize("objective", OBJECTIVES, ids=OBJECTIVES)
def test_portfolio(objective):

    portfolio = MeanVariancePortfolio(name=objective, objective=objective)
    portfolio.fit(asset_list=ASSET_LIST)
    portfolio_result = portfolio.evaluate()

    plot_portfolio = PlotPortfolio(portfolio_result)

    fig, ax = plot_portfolio.plot_returns()
    check_figure(fig, ax)

    fig, ax = plot_portfolio.plot_cum_returns()
    check_figure(fig, ax)

    fig, ax = plot_portfolio.plot_corr()
    check_figure(fig, ax)

    fig, ax = plot_portfolio.plot_cov()
    check_figure(fig, ax)

    fig, ax = plot_portfolio.plot_dist_returns()
    check_figure(fig, ax)

    fig, ax = plot_portfolio.plot_mean_sd()
    check_figure(fig, ax)

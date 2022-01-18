import pytest

from portfawn.portfolio.portfolio import MeanVariancePortfolio, PortfolioParams
from portfawn.plot.portfolio import PlotPortfolio
from utils import ASSET_LIST, OBJECTIVES


@pytest.mark.parametrize("objective", OBJECTIVES, ids=OBJECTIVES)
def test_portfolio(objective):

    portfolio_params = PortfolioParams(name=objective, objective=objective)
    portfolio = MeanVariancePortfolio(portfolio_params)
    portfolio_result = portfolio.run(asset_list=ASSET_LIST)

    plot_portfolio = PlotPortfolio(portfolio_result)

    fig, ax = plot_portfolio.plot_returns()
    fig.savefig("plots/portoflio_returns.png")

    fig, ax = plot_portfolio.plot_cum_returns()
    fig.savefig("plots/portoflio_cum_returns.png")

    fig, ax = plot_portfolio.plot_corr()
    fig.savefig("plots/portoflio_corr.png")

    fig, ax = plot_portfolio.plot_cov()
    fig.savefig("plots/portoflio_cov.png")

    fig, ax = plot_portfolio.plot_dist_returns()
    fig.savefig("plots/portoflio_dist_returns.png")

    fig, ax = plot_portfolio.plot_mean_sd()
    fig.savefig("plots/portoflio_mean_sd.png")

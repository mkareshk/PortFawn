import pytest

from portfawn.portfolio import Portfolio
from portfawn.portfolio import PlotPortfolio

asset_list = ["SPY", "GLD"]

quantum_objectives = ["BMOP"]
classic_objectives = ["EWP", "MRP", "MVP", "MSRP"]
objectives = quantum_objectives  # + classic_objectives


@pytest.mark.parametrize("objective", objectives, ids=objectives)
def test_portfolio(objective):
    portfolio = Portfolio(name=objective, objective=objective)
    portfolio_result = portfolio.run(asset_list=asset_list)

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

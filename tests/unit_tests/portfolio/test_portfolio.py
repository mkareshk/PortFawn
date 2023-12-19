import pytest
import dafin

from portfawn import MeanVariancePortfolio, PlotPortfolio
from tests.utils import ASSET_LIST, OBJECTIVES, check_figure
from portfawn import RiskModel, OptimizationModel


@pytest.mark.parametrize("objective", OBJECTIVES, ids=OBJECTIVES)
def test_portfolio(objective):
    # data
    data_instance = dafin.ReturnsData(ASSET_LIST)
    returns_data = data_instance.get_returns()

    # models
    optimization_model = OptimizationModel(objective=objective)

    # portfolio
    portfolio = MeanVariancePortfolio(
        name=objective, optimization_model=optimization_model
    )
    portfolio.fit(returns_assets=returns_data)
    portfolio_result = portfolio.evaluate(returns_data)

    # plot_portfolio = PlotPortfolio(
    #     portfolio_result, asset_list=portfolio_result.columns
    # )

    # fig, ax = plot_portfolio.plot_returns()
    # check_figure(fig, ax)

    # fig, ax = plot_portfolio.plot_cum_returns()
    # check_figure(fig, ax)

    # fig, ax = plot_portfolio.plot_corr()
    # check_figure(fig, ax)

    # fig, ax = plot_portfolio.plot_cov()
    # check_figure(fig, ax)

    # fig, ax = plot_portfolio.plot_dist_returns()
    # check_figure(fig, ax)

    # fig, ax = plot_portfolio.plot_mean_sd()
    # check_figure(fig, ax)

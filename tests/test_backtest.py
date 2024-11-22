import dafin
import matplotlib
import pytest

from portfawn import (
    BackTest,
    EquallyWeightedPortfolio,
    MeanVariancePortfolio,
    MeanVarianceRiskModel,
    OptimizationModel,
    RandomPortfolio,
)


@pytest.fixture
def sample_returns():
    assets_list = ["SPY", "GLD", "BND"]
    date_start = "2010-01-01"
    date_end = "2022-12-30"
    data_instance = dafin.ReturnsData(assets_list)
    return data_instance.get_returns(date_start=date_start, date_end=date_end)


@pytest.fixture
def valid_backtest():
    portfolio_list = [
        RandomPortfolio(),
        EquallyWeightedPortfolio(),
        MeanVariancePortfolio(),
    ]
    assets_list = ["SPY", "GLD", "BND"]
    date_start = "2023-01-01"
    date_end = "2023-12-31"
    fitting_days = 60
    evaluation_days = 20
    n_jobs = 1
    return BackTest(
        portfolio_list,
        assets_list,
        date_start,
        date_end,
        fitting_days,
        evaluation_days,
        n_jobs,
    )


def test_backtest_initialization(valid_backtest):
    bt = valid_backtest
    assert len(bt.analysis_windows) > 0, "Analysis windows should be created."
    assert bt.portfolio_list, "Portfolio list should not be empty."


@pytest.mark.parametrize(
    "date_start, date_end, fitting_days, evaluation_days",
    [
        ("2023-01-01", "2023-01-31", 30, 10),
        ("2023-01-01", "2023-12-31", 60, 30),
        ("2023-01-01", "2023-03-31", 10, 5),
    ],
)
def test_backtest_analysis_window(date_start, date_end, fitting_days, evaluation_days):
    portfolio_list = [EquallyWeightedPortfolio()]
    asset_list = ["SPY"]
    bt = BackTest(
        portfolio_list,
        asset_list,
        date_start,
        date_end,
        fitting_days,
        evaluation_days,
        n_jobs=1,
    )
    assert len(bt.analysis_windows) > 0, "Analysis windows should be created."


# Utility function to check if figure and axes are valid matplotlib objects
def check_figure(fig, ax):
    assert isinstance(
        fig, matplotlib.figure.Figure
    ), "Figure should be a valid matplotlib Figure object."
    assert isinstance(
        ax, matplotlib.axes.Axes
    ), "Axes should be a valid matplotlib Axes object."


# Test the BackTest functionality with multiple parameterized inputs
@pytest.mark.parametrize(
    "objectives, risk_free_rate, assets_list, date_start, date_end, fitting_days, evaluation_days, n_jobs, target_return, target_sd, weight_bound",
    [
        (
            ["MSRP", "MVP", "MSRP", "BMOP"],
            0.01,
            ["SPY", "BND"],
            "2023-01-01",
            "2023-12-31",
            30,
            10,
            1,
            0.15,
            0.2,
            (0.0, 1.0),
        ),
        (
            ["MVP"],
            0.02,
            ["BND", "SPY"],
            "2022-01-01",
            "2022-06-30",
            60,
            20,
            2,
            0.1,
            0.15,
            (0.0, 1.0),
        ),
    ],
)
def test_mean_var_backtest(
    objectives,
    risk_free_rate,
    assets_list,
    date_start,
    date_end,
    fitting_days,
    evaluation_days,
    n_jobs,
    target_return,
    target_sd,
    weight_bound,
):
    # Create a list of MeanVariancePortfolio instances with given objectives
    portfolio_list = [
        MeanVariancePortfolio(
            name=obj,
            optimization_model=OptimizationModel(
                objective=obj,
                risk_free_rate=risk_free_rate,
                optimization_params={
                    "target_return": target_return,
                    "target_sd": target_sd,
                    "weight_bound": weight_bound,
                },
            ),
        )
        for obj in objectives
    ]
    print(portfolio_list[0])

    # Initialize the BackTest object
    portfolio_backtest = BackTest(
        portfolio_list=portfolio_list,
        asset_list=assets_list,
        date_start=date_start,
        date_end=date_end,
        fitting_days=fitting_days,
        evaluation_days=evaluation_days,
        n_jobs=n_jobs,
    )

    # Run the backtest
    portfolio_backtest.run()

    # Test all plot functions
    fig, ax = portfolio_backtest.plot_returns()
    check_figure(fig, ax)

    fig, ax = portfolio_backtest.plot_cum_returns()
    check_figure(fig, ax)

    fig, ax = portfolio_backtest.plot_dist_returns()
    check_figure(fig, ax)

    fig, ax = portfolio_backtest.plot_corr()
    check_figure(fig, ax)

    fig, ax = portfolio_backtest.plot_cov()
    check_figure(fig, ax)

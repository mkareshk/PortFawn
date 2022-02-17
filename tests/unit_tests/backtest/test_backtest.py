from portfawn import BackTest
from portfawn import MeanVariancePortfolio
from ..utils import (
    ASSET_LIST,
    OBJECTIVES,
    DATE_START,
    DATE_END,
    FITTING_DATES,
    EVALUATION_DAYS,
    RISK_FREE_RATE,
    TARGET_RETURN,
    TARGET_SD,
    WEIGHT_BOUND,
    N_JOBS,
    check_figure,
)


def test_backtest():

    portfolio_list = []

    for obj in OBJECTIVES:
        portfolio = MeanVariancePortfolio(
            objective=obj,
            risk_free_rate=RISK_FREE_RATE,
            target_return=TARGET_RETURN,
            target_sd=TARGET_SD,
            weight_bound=WEIGHT_BOUND,
        )
        portfolio_list.append(portfolio)

    portfolio_backtest = BackTest(
        portfolio_list=portfolio_list,
        asset_list=ASSET_LIST,
        date_start=DATE_START,
        date_end=DATE_END,
        fitting_days=FITTING_DATES,
        evaluation_days=EVALUATION_DAYS,
        n_jobs=N_JOBS,
    )

    portfolio_backtest.run()

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

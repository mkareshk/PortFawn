from portfawn.portfolio.multi_portfolio import MultiPortfolio
from portfawn.plot.multi_portfolio import PlotMultiPortfolio

asset_list = ["SPY", "GLD"]

quantum_objectives = ["BMOP"]
classic_objectives = ["EWP", "MRP", "MVP", "MSRP"]
objectives = quantum_objectives + classic_objectives


def test_multi_portfolio():

    multi_portfolio = MultiPortfolio(name="multi", objectives_list=objectives)
    result = multi_portfolio.run(asset_list=asset_list)

    # fig, ax = multi_portfolio.plot_portfolio()
    plot_multi_portfolio = PlotMultiPortfolio(result)
    fig, ax = plot_multi_portfolio.plot_mean_sd()
    fig.savefig("plots/multi.png")

from portfawn.portfolio import MultiPortoflio

asset_list = ["SPY", "GLD"]

quantum_objectives = ["BMOP"]
classic_objectives = ["EWP", "MRP", "MVP", "MSRP"]
objectives = quantum_objectives + classic_objectives


def test_multi_portfolio():

    multi_portfolio = MultiPortoflio(name="multi", objectives_list=objectives)
    multi_portfolio.run(asset_list=asset_list)

    fig, ax = multi_portfolio.plot_portfolio()
    fig.savefig("plots/multi.png")

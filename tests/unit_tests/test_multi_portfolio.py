from portfawn.portfolio.multi_portfolio import MultiPortfolio
from portfawn.plot.multi_portfolio import PlotMultiPortfolio
from utils import ASSET_LIST, OBJECTIVES


def test_multi_portfolio():

    multi_portfolio = MultiPortfolio(name="multi", objectives_list=OBJECTIVES)
    result = multi_portfolio.run(asset_list=ASSET_LIST)

    plot_multi_portfolio = PlotMultiPortfolio(result)
    fig, ax = plot_multi_portfolio.plot_mean_sd()
    fig.savefig("plots/multi.png")

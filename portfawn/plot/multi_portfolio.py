import logging

from .plot import Plot

logger = logging.getLogger(__name__)


class PlotMultiPortfolio:
    def __init__(self, performance) -> None:
        self.performance = performance
        self.plot = Plot()

    def plot_mean_sd(self):
        fig, ax = self.plot.plot_scatter_portfolio_random(
            df_1=self.performance["market_mean_sd"],
            df_2=self.performance["portfolio_mean_sd"],
            df_3=self.performance["mean_sd_random"],
            title="",
            xlabel="Annualized Standard Deviation (%)",
            ylabel="Annualized Expected Returns (%)",
        )
        return fig, ax

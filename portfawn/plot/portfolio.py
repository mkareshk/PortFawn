import logging

from portfawn.plot import Plot
from portfawn.portfolio import RandomPortfolio

logger = logging.getLogger(__name__)


class PlotPortfolio:
    def __init__(self, performance) -> None:
        self.performance = performance
        self.asset_list = list(self.performance["asset_weights"].keys())
        self.portfolio_list = list(
            set(performance["returns"].columns) - set(self.asset_list)
        )
        self.plot = Plot()

    def plot_pie(self):
        fig, ax = self.plot.plot_pie(
            data_dict=self.performance["asset_weights"],
        )
        return fig, ax

    def plot_returns(self, resample):
        fig, ax = self.plot.plot_trend(
            df=self.performance["returns"].resample(resample).mean(),
            title=f"",
            xlabel="Date",
            ylabel="Daily Returns",
            legend=True,
            asset_list=self.asset_list,
            portfolio_list=self.portfolio_list,
        )
        return fig, ax

    def plot_cum_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.performance["cum_returns"],
            title="",
            xlabel="Date",
            ylabel="Cumulative Returns",
            asset_list=self.asset_list,
            portfolio_list=self.portfolio_list,
        )
        return fig, ax

    def plot_dist_returns(self):
        fig, ax = self.plot.plot_box(
            df=self.performance["returns"],
            title="",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns",
            yscale="symlog",
        )
        return fig, ax

    def plot_corr(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.performance["returns"],
            relation_type="corr",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.performance["returns"],
            relation_type="cov",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_mean_sd(
        self,
        annualized=True,
        fig=None,
        ax=None,
    ):
        if annualized:
            mv = self.performance["annualized_mean_sd"]
            xlabel = "Annualized Standard Deviation (%)"
            ylabel = "Annualized Expected Returns (%)"
        else:
            mv = self.performance["mean_sd"]
            xlabel = "Standard Deviation"
            ylabel = "Expected Returns"

        market_mean_sd = mv.loc[self.asset_list, :]
        portfolio_mean_sd = mv.loc[self.portfolio_list, :]
        random_mean_sd = RandomPortfolio(
            returns=self.performance["returns"].loc[:, self.asset_list],
            days_per_year=self.performance["days_per_year"],
            annualized=annualized,
        )

        fig, ax = self.plot.plot_scatter_portfolio_random(
            df_1=market_mean_sd,
            df_2=portfolio_mean_sd,
            df_3=random_mean_sd,
            title="",
            xlabel=xlabel,
            ylabel=ylabel,
        )

        return fig, ax

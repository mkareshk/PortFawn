import logging
from matplotlib.pyplot import yscale

import numpy as np
import pandas as pd

from portfawn.plot import Plot

logger = logging.getLogger(__name__)


class PlotPortfolio:
    def __init__(self, performance) -> None:
        self.performance = performance
        self.asset_list = list(self.performance["asset_weights"].keys())
        self.portfolio_list = list(performance["portfolio_returns"].columns)
        self.plot = Plot()

    def plot_pie(self):

        fig, ax = self.plot.plot_pie(
            data_dict=self.performance["asset_weights"],
        )
        return fig, ax

    def plot_returns(self, resample):

        fig, ax = self.plot.plot_trend(
            df=self.performance["portfolio_assets_returns"].resample(resample).mean(),
            title=f"",
            xlabel="Date",
            ylabel="Average Daily Returns",
            legend=True,
            asset_list=self.asset_list,
            portfolio_list=self.portfolio_list,
        )
        return fig, ax

    def plot_cum_returns(self):
        fig, ax = self.plot.plot_trend(
            df=self.performance["portfolio_assets_cum_returns"],
            title="",
            xlabel="Date",
            ylabel="Cumulative Returns",
            asset_list=self.asset_list,
            portfolio_list=self.portfolio_list,
        )
        return fig, ax

    def plot_dist_returns(self):
        fig, ax = self.plot.plot_box(
            df=self.performance["portfolio_assets_returns"],
            title="",
            xlabel="Portfolio Fitness",
            ylabel="Daily Returns",
            yscale="symlog",
        )
        return fig, ax

    def plot_corr(self):

        fig, ax = self.plot.plot_heatmap(
            df=self.performance["portfolio_assets_returns"],
            relation_type="corr",
            title="",
            annotate=True,
        )
        return fig, ax

    def plot_cov(self):
        fig, ax = self.plot.plot_heatmap(
            df=self.performance["portfolio_assets_returns"],
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

        market_mean_sd = self.performance["market_mean_sd"].copy()
        portfolio_mean_sd = self.performance["portfolio_mean_sd"].copy()
        random_mean_sd = self.random_portfolio(self.performance["asset_returns"])

        annualized_days = self.performance["portfolio_config"]["annualized_days"]

        if annualized:
            market_mean_sd["mean"] *= annualized_days
            market_mean_sd["sd"] *= np.sqrt(annualized_days)
            portfolio_mean_sd["mean"] *= annualized_days
            portfolio_mean_sd["sd"] *= np.sqrt(annualized_days)
            random_mean_sd["mean"] *= annualized_days
            random_mean_sd["sd"] *= np.sqrt(annualized_days)

        fig, ax = self.plot.plot_scatter_portfolio_random(
            df_1=market_mean_sd,
            df_2=portfolio_mean_sd,
            df_3=random_mean_sd,
            title="",
            xlabel="Volatility (SD)",
            ylabel="Expected Returns",
        )

        return fig, ax

    def random_portfolio(self, asset_returns):
        n = 1000
        returns_np = asset_returns.to_numpy()
        cov = asset_returns.cov().to_numpy()
        r_list = []
        for i in range(n):
            w_rand = np.random.random((1, cov.shape[0]))
            w_rand = w_rand / w_rand.sum()
            r = returns_np.dot(w_rand.T).mean()
            c = np.sqrt(w_rand.dot(cov).dot(w_rand.T))[0][0]
            r_list.append({"mean": r, "sd": c})
        return pd.DataFrame(r_list)

from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# general configuration for matplotlib
params = {
    "font.family": "serif",
    "legend.fontsize": "large",
    "figure.figsize": (10, 7),
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
pylab.rcParams.update(params)


class Plot:
    def __init__(self, asset_num=0, path_results="", plot_type="portfolio"):

        # parameters
        self.asset_num = asset_num
        self.path_results = path_results
        self.plot_type = plot_type

        # styles
        self.color_base_list = [
            "tab:blue",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:cyan",
            "tab:orange",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tan",
            "lime",
            "indigo",
        ]
        self.line_style_base = [":", "-.", "--"]
        self.linewidth_base = 0.5
        self.linewidth_portfolio = 2.0
        self.alpha_base = 0.5
        self.linecolor_list = 100 * self.color_base_list
        self.linestyle_list = 100 * self.line_style_base

        # log
        self.logger = logging.getLogger(__name__)

    def plot_box(self, returns, title="", xlabel="", ylabel="", filename=""):
        returns.plot.box(
            showfliers=False,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            sym=None,
            patch_artist=True,
            boxprops=dict(facecolor="royalblue", color="black"),
            medianprops=dict(linestyle="-", linewidth=2.5, color="khaki"),
        )
        plt.xlabel(xlabel)
        plt.grid(True, axis="y")
        locs, labels = plt.xticks()
        plt.xticks(locs, self.normalize_label(labels), rotation=45)
        plt.tight_layout()
        plt.savefig(self.path_results / Path(filename + ".png"))
        plt.savefig(self.path_results / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    def plot_heatmap(self, relations, relation_type, title="", filename=""):

        if relation_type == "corr":
            annot_fmt = "0.2f"
            shift = 0
            vmin, vmax = -1, 1
            relations = relations.iloc[:, 0:-1]
        elif relation_type == "cov":
            annot_fmt = "0.5f"
            shift = 1
            vmin, vmax = relations.min().min(), relations.max().max()
        mask = np.zeros_like(relations)
        mask[np.triu_indices_from(mask, k=shift)] = True
        sns.heatmap(
            relations,
            mask=mask,
            cmap="RdYlGn",
            xticklabels=relations.columns,
            yticklabels=relations.columns,
            annot=True,
            fmt=annot_fmt,
            annot_kws={"fontsize": 14},
            vmin=vmin,
            vmax=vmax,
        )

        if relation_type == "corr":
            locs, labels = plt.xticks()
            plt.xticks(locs[0:], labels[0:])
            locs, labels = plt.yticks()
            plt.yticks(locs[1:], labels[1:])

        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.path_results / Path(filename + ".png"))
        plt.savefig(self.path_results / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    def plot_trend(self, returns, title="", xlabel="", ylabel="", filename=""):
        cols_portfolio = [c for c in returns.columns if c.count("portfolio_") > 0]
        if len(cols_portfolio) > 0:
            cols_assets = [c for c in returns.columns if c.count("asset_") > 0]
        else:
            cols_assets = returns.columns

        returns_portfolio = returns.drop(cols_assets, axis=1)
        returns_assets = returns.drop(cols_portfolio, axis=1)

        if len(cols_portfolio) > 0:  # DF with both assets and portfolios
            returns_portfolio.plot(
                # ax=ax,
                # title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                color=self.linecolor_list[0 : len(cols_portfolio)],
                # style=self.linestyle_list[0 : len(cols_portfolio)],
                linewidth=self.linewidth_portfolio,
                alpha=1.0,
            )
        else:  # asset only DF
            returns.plot(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                color=self.linecolor_list[0 : self.asset_num],
                linewidth=2.0,
            )
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        plt.xticks(rotation=45)
        plt.legend(current_handles, self.normalize_label(current_labels))
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.path_results / Path(filename + ".png"))
        plt.savefig(self.path_results / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    def plot_bar(
        self,
        df,
        yscale="linear",
        title="",
        legend_title="",
        xlabel="",
        ylabel="",
        filename="",
    ):

        df.plot.bar()
        plt.grid(True, axis="y")
        locs, labels = plt.xticks()
        plt.xticks(locs, self.normalize_label(labels), rotation=45)
        plt.yscale(yscale)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title=legend_title
        )

        plt.tight_layout()
        plt.savefig(self.path_results / Path(filename + ".png"))
        plt.savefig(self.path_results / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    @staticmethod
    def normalize_label(labels):

        if not labels or len(labels) < 1:
            return

        if not isinstance(labels[0], str):
            labels = [l.get_text() for l in labels]

        possible_extra = ["portfolio_sharpe_ratio", "portfolio_total_return"]
        labels_new = []
        for l in labels:
            t = l
            for p in possible_extra:
                t = t.replace(p, "")
            if len(t) < 5:  # assets
                t = t.replace("_", " ").upper()
            else:  # portfolio
                t = t.replace("_", " ").title() + " Portfolio"
            labels_new.append(t)
        return labels_new

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# general configuration for matplotlib
params = {
    "font.family": "serif",
    "legend.fontsize": "large",
    "figure.figsize": (8, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
pylab.rcParams.update(params)


class Plot:
    def __init__(self, path_plot=""):

        # parameters
        self.path_plot = path_plot

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
        plt.savefig(self.path_plot / Path(filename + ".png"))
        plt.savefig(self.path_plot / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    def plot_heatmap(self, returns, relation_type, title="", filename=""):

        if relation_type == "corr":
            relations = returns.corr()
            annot_fmt = "0.2f"
            shift = 0
            vmin, vmax = -1, 1
        elif relation_type == "cov":
            relations = returns.cov()
            annot_fmt = "0.2f"
            shift = 0
            vmin, vmax = relations.min().min(), relations.max().max()

        sns.heatmap(
            relations,
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
        plt.savefig(self.path_plot / Path(filename + ".png"))
        plt.savefig(self.path_plot / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    def plot_trend(self, returns, title="", xlabel="", ylabel="", filename=""):

        returns.plot(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=self.linecolor_list,
            linewidth=self.linewidth_portfolio,
            alpha=1.0,
        )

        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        plt.xticks(rotation=45)
        plt.legend(
            current_handles,
            self.normalize_label(current_labels),
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
        )
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.path_plot / Path(filename + ".png"))
        plt.savefig(self.path_plot / Path(filename + ".pdf"))
        plt.clf()
        plt.close("all")

    def plot_bar(
        self,
        returns,
        yscale="linear",
        title="",
        legend_title="",
        xlabel="",
        ylabel="",
        filename="",
    ):

        returns.plot.bar()
        plt.grid(True, axis="y")
        locs, labels = plt.xticks()
        plt.xticks(locs, labels, rotation=45)
        plt.yscale(yscale)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title=legend_title
        )

        plt.tight_layout()
        plt.savefig(self.path_plot / Path(filename + ".png"))
        plt.savefig(self.path_plot / Path(filename + ".pdf"))
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
                t = t.replace("_", " ").title()
            labels_new.append(t)
        return labels_new


def demo():

    date = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]

    a = [1.5, 2, 3, 4.5, 5]
    b = [1, -1, 2, 7, 4]
    c = [2, -1, 3, -1, 2]
    d = [5, 4, 3, 2, 1]
    e = [0, 2, 4, 6, 8]

    data_df = pd.DataFrame(index=date)
    data_df["a"] = a
    data_df["b"] = b
    data_df["c"] = c
    data_df["d"] = d
    data_df["e"] = e

    plot = Plot(path_plot="temp")

    plot.plot_box(
        returns=data_df,
        title="my_title",
        xlabel="my_x",
        ylabel="my_y",
        filename="my_boxplot",
    )

    plot.plot_heatmap(
        data_df, relation_type="corr", title="my_corr", filename="my_corr"
    )
    plot.plot_heatmap(data_df, relation_type="cov", title="my_cov", filename="my_cov")

    plot.plot_trend(
        returns=data_df,
        title="my_title",
        xlabel="my_x",
        ylabel="my_y",
        filename="my_boxtrend",
    )

    plot.plot_bar(
        returns=data_df,
        title="my_title",
        xlabel="my_x",
        ylabel="my_y",
        filename="my_bar",
    )

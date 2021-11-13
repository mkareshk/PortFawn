import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from scipy.spatial import ConvexHull

sns.set()
sns.set_style("whitegrid")

# general configuration for matplotlib
params = {
    "font.family": "serif",
    "legend.fontsize": "large",
    "figure.figsize": (15, 8),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
pylab.rcParams.update(params)


class Plot:
    def __init__(self):

        # log
        self.logger = logging.getLogger(__name__)

    def plot_box(self, df, title="", xlabel="", ylabel="", figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        df.plot.box(
            showfliers=False,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            sym=None,
            patch_artist=True,
            boxprops=dict(facecolor="royalblue", color="black"),
            medianprops=dict(linestyle="-", linewidth=2.5, color="khaki"),
            ax=ax,
        )

        t = np.fabs(df.to_numpy())
        t = t[t > 0]
        ax.set_yscale("symlog", linthresh=min(t))
        import matplotlib

        # for axis in [ax.xaxis, ax.yaxis]:
        #     axis.set_major_formatter(LogFormatter())

        ax.set_xlabel(xlabel)
        plt.grid(True, axis="y")
        locs, labels = plt.xticks()
        plt.xticks(locs, labels, rotation=45)
        fig.tight_layout()

        return fig, ax

    def plot_heatmap(self, df, relation_type, title="", annotate=True, figsize=(15, 8)):

        fig, ax = plt.subplots(figsize=figsize)

        if relation_type == "corr":
            relations = df.corr()
            annot_fmt = "0.1f"
            shift = 1
            vmin, vmax = -1, 1
        elif relation_type == "cov":
            relations = df.cov()
            annot_fmt = "0.1g"
            shift = 1
            vmin, vmax = relations.min().min(), relations.max().max()

        mask = np.zeros_like(relations)
        mask[np.triu_indices_from(mask, k=shift)] = True

        sns.heatmap(
            relations,
            cmap="RdYlGn",
            mask=mask,
            xticklabels=relations.columns,
            yticklabels=relations.columns,
            annot=annotate,
            fmt=annot_fmt,
            annot_kws={"fontsize": 14},
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        if relation_type == "corr":
            locs, labels = plt.xticks()
            plt.xticks(locs, labels)
            locs, labels = plt.yticks()
            plt.yticks(locs, labels)

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(title)
        fig.tight_layout()

        return fig, ax

    def plot_trend(
        self, df, title="", xlabel="", ylabel="", figsize=(15, 8), alpha=1, legend=True
    ):
        fig, ax = plt.subplots(figsize=figsize)

        df.plot(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            linewidth=2,
            alpha=alpha,
            ax=ax,
        )
        if legend:
            current_handles, current_labels = plt.gca().get_legend_handles_labels()
            # plt.xticks(rotation=45)
            plt.legend(
                current_handles,
                current_labels,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
            )
        else:
            ax.get_legend().remove()
        plt.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_bar(
        self, df, yscale="linear", title="", xlabel="", ylabel="", figsize=(15, 8)
    ):
        fig, ax = plt.subplots(figsize=figsize)
        df.plot.bar(ax=ax, legend=False)
        plt.grid(True, axis="y")
        locs, labels = plt.xticks()
        plt.xticks(locs, labels, rotation=45)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        fig.tight_layout()

        return fig, ax

    def plot_scatter(
        self,
        df,
        title="",
        xlabel="",
        ylabel="",
        figsize=(15, 8),
        colour="tab:blue",
        fig=None,
        ax=None,
    ):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)

        df.plot.scatter(x="std", y="mean", c=colour, ax=ax, s=200, alpha=0.8)

        x_min, x_max = df["std"].min(), df["std"].max()
        x_diff = x_max - x_min
        y_min, y_max = df["mean"].min(), df["mean"].max()
        y_diff = y_max - y_min

        for i, point in df.iterrows():
            ax.text(
                point["std"] - x_diff * 0.1,
                point["mean"] + y_diff * 0.03,
                i,
                fontsize=14,
            )

        plt.grid(True, axis="y")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.set_xlim(left=x_min - 0.2 * x_diff, right=x_max + 0.2 * x_diff)
        ax.set_ylim(bottom=y_min - 0.2 * y_diff, top=y_max + 0.2 * y_diff)
        fig.tight_layout()

        return fig, ax

    def plot_scatter_portfolio(
        self,
        df_1,
        df_2,
        title="",
        xlabel="",
        ylabel="",
        figsize=(15, 8),
        colours=["tab:blue", "tab:red"],
    ):
        fig, ax = plt.subplots(figsize=figsize)

        df_1.plot.scatter(x="std", y="mean", c=colours[0], ax=ax, s=200, alpha=0.8)
        df_2.plot.scatter(x="std", y="mean", c=colours[1], ax=ax, s=200, alpha=0.8)

        x_min, x_max = df_1["std"].min(), df_1["std"].max()
        x_diff = x_max - x_min
        y_min, y_max = df_1["mean"].min(), df_1["mean"].max()
        y_diff = y_max - y_min

        for i, point in df_1.iterrows():
            ax.text(
                point["std"] - x_diff * 0.05,
                point["mean"] + y_diff * 0.05,
                i,
                fontsize=14,
            )
        for i, point in df_2.iterrows():
            ax.text(
                point["std"] - x_diff * 0.05,
                point["mean"] + y_diff * 0.05,
                i,
                fontsize=14,
            )
        plt.grid(True, axis="y")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.set_xlim(left=x_min - 0.2 * x_diff, right=x_max + 0.2 * x_diff)
        ax.set_ylim(bottom=y_min - 0.2 * y_diff, top=y_max + 0.2 * y_diff)
        fig.tight_layout()

        return fig, ax

    def plot_scatter_portfolio_random(
        self,
        df_1,
        df_2,
        df_3,
        title="",
        xlabel="",
        ylabel="",
        figsize=(15, 8),
        colours=["tab:blue", "tab:red", "tab:green"],
    ):
        fig, ax = plt.subplots(figsize=figsize)

        conc = pd.concat([df_1, df_2, df_3], axis=0).dropna()
        # hull = ConvexHull(conc.values)

        l = conc.iloc[np.argmin(conc["std"]), :]
        x = conc.values[:, 1]
        y = conc.values[:, 0]
        hull = ConvexHull(conc.values)

        vertices = [v for v in hull.vertices[3:] if y[v] > l["mean"]]

        plt.plot(x[vertices], y[vertices], "k--", linewidth=2, alpha=0.8)

        x_min, x_max = df_1["std"].min(), df_1["std"].max()
        x_diff = x_max - x_min
        y_min, y_max = df_1["mean"].min(), df_1["mean"].max()
        y_diff = y_max - y_min

        df_1.plot.scatter(x="std", y="mean", c=colours[0], ax=ax, s=200, alpha=0.8)
        df_2.plot.scatter(x="std", y="mean", c=colours[1], ax=ax, s=200, alpha=0.8)
        df_3.plot.scatter(x="std", y="mean", c=colours[2], ax=ax, s=1, alpha=0.01)

        for i, point in df_1.iterrows():
            ax.text(
                point["std"] - x_diff * 0.05,
                point["mean"] + y_diff * 0.05,
                i,
                fontsize=14,
            )
        for i, point in df_2.iterrows():
            ax.text(
                point["std"] - x_diff * 0.05,
                point["mean"] + y_diff * 0.05,
                i,
                fontsize=14,
            )
        plt.grid(True, axis="y")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.set_xlim(left=x_min - 0.3 * x_diff, right=x_max + 0.3 * x_diff)
        ax.set_ylim(bottom=y_min - 0.3 * y_diff, top=y_max + 0.3 * y_diff)
        fig.tight_layout()

        return fig, ax

import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

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

        ax.set_xlabel(xlabel)
        plt.grid(True, axis="y")
        locs, labels = plt.xticks()
        plt.xticks(locs, labels, rotation=45)
        fig.tight_layout()

        return fig

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

        return fig

    def plot_trend(self, df, title="", xlabel="", ylabel="", figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize)

        df.plot(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            linewidth=2,
            ax=ax,
        )

        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        plt.xticks(rotation=45)
        plt.legend(
            current_handles,
            current_labels,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
        )
        plt.grid(True)
        fig.tight_layout()

        return fig

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

        return fig

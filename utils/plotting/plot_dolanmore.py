import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("science")


def dolan_more(
    merged, title="Dolan-More curve", dataset_col="Dataset", our_method_name="Cohortney"
):
    df = merged.copy(deep=True)
    df["best_performance"] = np.array(df.set_index(dataset_col).min(axis=1))
    betas = pd.DataFrame(df.set_index(dataset_col))
    for i in df.set_index(dataset_col).columns:
        betas[i] = (
            df.set_index(dataset_col)[i] / df.set_index(dataset_col)["best_performance"]
        )
    max_beta = betas[betas.columns[:-1]].replace(np.inf, 0).max().max()
    space = np.linspace(1, max_beta, 100)
    betas = betas.replace(np.inf, max_beta)

    curves = []
    methods = []
    for i in df.set_index(dataset_col).columns[:-1]:
        curve = []
        methods.append(i)
        for sp in space:
            curve.append(np.sum(betas[i] <= sp) / df.shape[0])
        curves.append(curve)

    f = plt.figure(figsize=(7, 5), dpi=180)
    plt.title(title)
    i = 0
    for method, curve in zip(methods, curves):
        if method == our_method_name:
            plt.plot(
                space,
                curve,
                label=method,
                linewidth=3,
                c=sns.color_palette("bright", 8)[4],
                linestyle="--",
            )
        else:
            if i == 4:
                i += 1
            plt.plot(space, curve, label=method, c=sns.color_palette("bright", 10)[i])
        i += 1
    plt.ylabel("Proportion of datasets")
    plt.xlabel(r"$\beta$")
    plt.legend()
    # plt.show()
    return f


def dolan_more_mib(
    merged, title="Dolan-More curve", dataset_col="Dataset", our_method_name="Cohortney"
):
    df = merged.copy(deep=True)
    df["best_performance"] = np.array(df.set_index(dataset_col).max(axis=1))
    betas = pd.DataFrame(df.set_index(dataset_col))
    for i in df.set_index(dataset_col).columns:
        betas[i] = (
            df.set_index(dataset_col)["best_performance"] / df.set_index(dataset_col)[i]
        )
    max_beta = betas[betas.columns[:-1]].replace(np.inf, 0).max().max()
    space = np.linspace(1, max_beta, 100)
    betas = betas.replace(np.inf, max_beta)

    curves = []
    methods = []
    for i in df.set_index(dataset_col).columns[:-1]:
        curve = []
        methods.append(i)
        for sp in space:
            curve.append(np.sum(betas[i] <= sp) / df.shape[0])
        curves.append(curve)

    f = plt.figure(figsize=(7, 5), dpi=180)
    plt.title(title)
    i = 0
    for method, curve in zip(methods, curves):
        if method == our_method_name:
            plt.plot(
                space,
                curve,
                label=method,
                linewidth=3,
                c=sns.color_palette("bright", 8)[4],
                linestyle="--",
            )
        else:
            if i == 4:
                i += 1
            plt.plot(space, curve, label=method, c=sns.color_palette("bright", 10)[i])
        i += 1
    plt.ylabel("Proportion of datasets")
    plt.xlabel(r"$\beta$")
    plt.legend()
    # plt.show()
    return f


if __name__ == "__main__":
    # metrics are purity, adjusted information score
    naming_dict = {
        "purity": "purities",
        "adjusted mutual info score": "adj_mut_info_score",
        "adjusted rand score": "adj_rand_score",
        "v-measure score": "v_meas_score",
        "fowlkes mallows score": "f_m_score",
    }
    metric = "adjusted mutual info score"
    title = "Dolan-More Curve based on " + metric
    for sumtrue in ["", "_sum"]:
        res_df = pd.read_csv(
            naming_dict[metric] + sumtrue + "_dm_res.csv", index_col=False
        )
        res_df.rename(
            columns={
                "COHORTNEY": "Cohortney",
                "DMHP": "Zhu",
                "K-means": "K-means partitions",
                "K-means0": "K-means tsfresh",
                "GMM": "GMM tsfresh",
            },
            inplace=True,
        )
        cols = [c for i, c in enumerate(res_df.columns) if i > 0]
        res_df = res_df[cols]
        f = dolan_more_mib(res_df, title=title)
        f.savefig(metric + sumtrue + "_dolanmore.pdf", dpi=400, bbox_inches="tight")

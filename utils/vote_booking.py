import os

import pandas as pd

# iterate over diff columns experiments folders
n_runs = 1
for i in range(0, n_runs):
    # df with columns to vote
    devcl = pd.read_csv(
        os.path.join(
            "experiments/booking", "deviceclass", "exp_" + str(i), "compareclusters.csv"
        )
    )
    diffinout = pd.read_csv(
        os.path.join(
            "experiments/booking", "diffinout", "exp_" + str(i), "compareclusters.csv"
        )
    )
    diffcheckin = pd.read_csv(
        os.path.join(
            "experiments/booking", "diffcheckin", "exp_" + str(i), "compareclusters.csv"
        )
    )
    # csv with results
    res_df = pd.DataFrame()
    res_df["seqlength"] = devcl["seqlength"].values
    # ground truth
    res_df["cluster_id"] = devcl["cluster_id"].values
    # cohortney labels voting
    res_df["coh_cluster_1"] = devcl["coh_cluster"].values
    res_df["coh_cluster_2"] = diffinout["coh_cluster"].values
    res_df["coh_cluster_3"] = diffcheckin["coh_cluster"].values
    res_df["time"] = (
        devcl["time"].values + diffinout["time"].values + diffcheckin["time"].values
    )
    # voting - inferred cluster is most frequent
    res_df["coh_cluster"] = (
        res_df[["coh_cluster_1", "coh_cluster_2", "coh_cluster_3"]]
        .mode(axis=1)
        .iloc[:, 0]
    )
    # tsfresh features with standard cluster algos
    methods = ["kmeans", "gmm"]
    for m in methods:
        res_df[m + "_time"] = (
            devcl[m + "_time"].values
            + diffinout[m + "_time"].values
            + diffcheckin[m + "_time"].values
        )
        res_df[m + "_cluster_1"] = devcl[m + "_clusters"].values
        res_df[m + "_cluster_2"] = diffinout[m + "_clusters"].values
        res_df[m + "_cluster_3"] = diffcheckin[m + "_clusters"].values
        res_df[m + "_clusters"] = (
            res_df[[m + "_cluster_1", m + "_cluster_2", m + "_cluster_3"]]
            .mode(axis=1)
            .iloc[:, 0]
        )

    res_df.to_csv(
        os.path.join("experiments/booking", "exp_" + str(i), "compare_clusters.csv")
    )

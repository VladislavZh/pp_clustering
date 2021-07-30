import argparse
import collections
import glob
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from metrics import purity
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     fowlkes_mallows_score,
                                     normalized_mutual_info_score,
                                     v_measure_score)


def metrics_map(metrics_name: str):
    """
    Maps metrics name from results dictionary to metrics function
    """
    metrics_dict = {
        "purities": purity,
        "adj_mut_info_score": adjusted_mutual_info_score,
        "adj_rand_score": adjusted_rand_score,
        "v_meas_score": v_measure_score,
        "f_m_score": fowlkes_mallows_score,
    }

    return metrics_dict[metrics_name]


def cohortney_tsfresh_stats(dataset: str, methods: List[str]):
    """
    Function to obtain summary statistics of Cohortney/Tsfresh on dataset
    """
    exp_folder = os.path.join("experiments", dataset)
    experiments = glob.glob(exp_folder + "/exp_*")
    res_dict = collections.defaultdict(dict)
    mtrcs = [
        "purities",
        "adj_mut_info_score",
        "adj_rand_score",
        "v_meas_score",
        "f_m_score",
    ]
    for m in methods:
        res_dict[m]["time"] = 0
        res_dict[m]["train_time"] = 0
        for metr in mtrcs:
            res_dict[m][metr] = []

    other_methods = ["zhu", "kmeans_ps", "kshape"]
    for om in other_methods:
        for metr in mtrcs:
            res_dict[om][metr] = []

    # iterating over experiments resuls
    n_runs = 0
    for exp in experiments:

        clusters = os.path.join(exp, "compare_clusters.csv")
        if os.path.exists(clusters):
            df = pd.read_csv(clusters)
            true_labels = df["cluster_id"].to_list()
            for m in methods:
                # cohortney
                if m == "cohortney":
                    pred_labels = df["coh_cluster"].to_list()
                    res_dict["cohortney"]["time"] += df["time"][0]
                else:
                    pred_labels = df[m + "_clusters"].to_list()
                    res_dict[m]["time"] += df[m + "_time"][0]
                for metrics_name in mtrcs:
                    single_score = metrics_map(metrics_name)(true_labels, pred_labels)
                    res_dict[m][metrics_name].append(single_score)

        # training time
        res_file = os.path.join(exp, "results.pkl")
        if os.path.exists(res_file):
            n_runs += 1
            with open(res_file, "rb") as f:
                res_list = pickle.load(f)
            res_dict["cohortney"]["train_time"] += res_list[-1][4]

    # zhu - dmhp
    zhu_exp_folder = os.path.join("experiments", "Zhu_experiments", dataset)
    experiments = glob.glob(zhu_exp_folder + "/exp_*")
    if len(experiments) == 0:
        res_dict["zhu"]["purities"] = [0.0]
    else:
        for exp in experiments:
            clusters = os.path.join(exp, "inferredclusters.csv")
            if os.path.exists(clusters):
                df = pd.read_csv(clusters)
                true_labels = df["cluster_id"].to_list()
                pred_labels = df["zhu_cluster"].to_list()
                for metrics_name in mtrcs:
                    single_score = metrics_map(metrics_name)(true_labels, pred_labels)
                    res_dict["zhu"][metrics_name].append(single_score)

    # kmeans kshape
    k_exp_folder = os.path.join("experiments", "Kmeans_Kshape_experiments", dataset)
    clusters = os.path.join(k_exp_folder, "inferredclusters.csv")
    if os.path.exists(clusters):
        df = pd.read_csv(clusters)
        true_labels = df["cluster_id"].to_list()
        kmeans_labels = df["kmeans_cluster"].to_list()
        for metrics_name in mtrcs:
            single_score = metrics_map(metrics_name)(true_labels, kmeans_labels)
            res_dict["kmeans_ps"][metrics_name] = [single_score]
        kshape_labels = df["kshape_cluster"].to_list()
        for metrics_name in mtrcs:
            single_score = metrics_map(metrics_name)(true_labels, kshape_labels)
            res_dict["kshape"][metrics_name] = [single_score]

    res_dict["n_runs"] = n_runs
    res_dict["n_clusters"] = len(np.unique(np.array(true_labels)))

    return res_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="exp_K2_C5")
    args = parser.parse_args()
    methods = ["cohortney", "kmeans", "gmm"]
    res_dict = cohortney_tsfresh_stats(args.dataset, methods)
    print("dataset", args.dataset)
    print("number of runs", res_dict["n_runs"])
    print("number of clusters", res_dict["n_clusters"])
    for m in methods:
        print(m)
        print("mean alg time", res_dict[m]["time"] / res_dict["n_runs"])
        print("mean purity", np.mean(np.array(res_dict[m]["purities"])))
        print("stdev purity", np.std(np.array(res_dict[m]["purities"])))

    print("dmhp", np.mean(np.array(res_dict["zhu"]["purities"])))
    print("kmeans", np.mean(np.array(res_dict["kmeans_ps"]["purities"])))
    print("kshape", np.mean(np.array(res_dict["kshape"]["purities"])))

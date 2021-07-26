import argparse
import collections
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch

from metrics import purity
from typing import List

def cohortney_tsfresh_stats(dataset: str, methods: List[str]):
    """
        Function to obtain summary statistics of Cohortney/Tsfresh on dataset
    """
    exp_folder = os.path.join("experiments", dataset)
    experiments = glob.glob(exp_folder + "/exp_*")
    res_dict = collections.defaultdict(dict)
    for m in methods:
        res_dict[m]["time"] = 0
        res_dict[m]["train_time"] = 0
        res_dict[m]["purities"] = []

    # iterating over experiments resuls
    n_runs = 0
    for exp in experiments:
        clusters = os.path.join(exp, "compare_clusters.csv")
        if os.path.exists(clusters):
            n_runs += 1
            df = pd.read_csv(clusters)
            true_labels = df["cluster_id"].to_list()
            for m in methods:
                # cohortney
                if m == "cohortney":
                    labels = df["coh_cluster"].to_list()
                    res_dict["cohortney"]["time"] += df["time"][0]
                    pur = purity(torch.FloatTensor(true_labels), torch.FloatTensor(labels))
                    res_dict["cohortney"]["purities"].append(pur)
                else:
                    labels = df[m+"_clusters"].to_list()
                    res_dict[m]["time"] += df[m+"_time"][0]
                    pur = purity(torch.FloatTensor(true_labels), torch.FloatTensor(labels))
                    res_dict[m]["purities"].append(pur)
        # training time 
        res_file = os.path.join(exp, "results.pkl")
        if os.path.exists(res_file):
            with open(res_file, "rb") as f:
                res_list = pickle.load(f)
            res_dict["cohortney"]["train_time"] += res_list[-1][4]

    res_dict["n_runs"] = n_runs
    res_dict["n_clusters"] = len(np.unique(np.array(true_labels)))

    return res_dict

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="K2_C5")
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

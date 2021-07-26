import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tsfresh
from sklearn import mixture
from sklearn.cluster import KMeans
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

from utils.data_preprocessor import get_dataset


def tsfresh_clusters(path_to_data, n_classes, n_steps, n_clusters, col_to_select=None):
    """
    creates tsfresh features and clusterizes dataset with standard algorithms
    """
    t1 = time.time()
    data, target = get_dataset(path_to_data, n_classes, n_steps, col_to_select)
    n_classes += 1
    len_data = data.shape[0]
    data_divided = []
    for i in range(n_classes):
        data_divided.append(data[:, :, i].reshape(-1))
    to_extract = []
    for i in range(n_classes):
        ids = np.arange(len_data).repeat(n_steps)
        tmp = np.vstack((ids, data_divided[i]))
        tmp = tmp.T
        to_extract.append(pd.DataFrame(data=tmp, columns=["id", "value"]))
    tfs = []
    # parameters of tsfresh features extraction
    settings = ComprehensiveFCParameters()  # or MinimalFCParameters()
    for i in range(n_classes):
        tf = tsfresh.extract_features(
            to_extract[i], column_id="id", default_fc_parameters=settings
        )
        tfs.append(tf)
    data_feat = pd.concat(
        [tfs[i].reindex(tfs[0].index) for i in range(n_classes)], axis=1
    )
    data_feat.fillna(0, inplace=True)
    data_feat.replace([np.inf, -np.inf], 0, inplace=True)
    # dump tsfresh features to temporary csv
    path_to_experiments = os.path.join("experiments", path_to_data.split("/")[-1])
    data_feat.to_csv(os.path.join(path_to_experiments, "tsfreshfeatures.csv"))
    # obtaining clusters
    res_dict = {}
    res_dict["data"] = path_to_data
    res_dict["n_classes"] = n_classes
    res_dict["n_steps"] = n_steps
    res_dict["n_clusters"] = n_clusters
    t2 = time.time()
    res_dict["kmeans"] = {
        "clusters": [],
        "time": round(t2 - t1, 5),
    }
    res_dict["gmm"] = {
        "clusters": [],
        "time": round(t2 - t1, 5),
    }
    # iterating over large csv
    data_iter = pd.read_csv(
        os.path.join(path_to_experiments, "tsfreshfeatures.csv"),
        iterator=True,
        chunksize=15000,
    )

    for chunk in data_iter:
        t2 = time.time()
        print("KMeans started")
        model = KMeans(n_clusters=n_clusters, max_iter=200)
        model.fit(chunk)
        kmeans_clusters = model.predict(chunk)
        t3 = time.time()
        res_dict["kmeans"]["time"] += round(t3 - t2, 5)
        res_dict["kmeans"]["clusters"].extend(kmeans_clusters)

        print("GMM started")
        g = mixture.GaussianMixture(
            n_components=n_clusters, covariance_type="diag", max_iter=100
        )
        g.fit(chunk)
        gmm_clusters = g.predict(chunk)
        t4 = time.time()
        res_dict["gmm"]["time"] += round(t4 - t3, 5)
        res_dict["gmm"]["clusters"].extend(gmm_clusters)

    return res_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sin_K5_C5")
    parser.add_argument("--col_to_select", type=str)
    args = parser.parse_args()
    # path to dataset
    data_path = os.path.join("data", args.dataset)
    # path to experiments settings
    exper_path = os.path.join("experiments", args.dataset, "exp_0")
    with open(os.path.join(exper_path, "args.json")) as json_file:
        config = json.load(json_file)
    n_steps = config["n_steps"]
    n_classes = config["n_classes"]
    n_clusters = config["true_clusters"]

    res_dict = tsfresh_clusters(
        data_path, n_classes, n_steps, n_clusters, args.col_to_select
    )
    # saving results
    res_df = pd.read_csv(os.path.join(data_path, "clusters.csv"))
    res_df["seqlength"] = 0
    csvfiles = sorted(os.listdir(data_path))
    for index, row in res_df.iterrows():
        seq_df = pd.read_csv(os.path.join(data_path, csvfiles[index]))
        res_df.at[index, "seqlength"] = len(seq_df)
    #res_df = pd.read_csv(os.path.join(exper_path, "cohortney_clusters.csv"))
    methods = ["kmeans", "gmm"]
    for m in methods:
        res_df[m + "_clusters"] = np.array(res_dict[m]["clusters"])
        res_df[m + "_time"] = res_dict[m]["time"]

    #save_path = os.path.join(exper_path, "compare_clusters.csv")
    save_path = os.path.join(exper_path, "tsfresh_clusters.csv")
    res_df.drop(
        res_df.columns[res_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    res_df.to_csv(save_path)

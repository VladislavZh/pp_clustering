import dropbox
import os
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

from data_preprocessor import dropbox_download, stopwatch

DROPBOX_TOKEN = "AS74Amc6RgcAAAAAAAAAAZJXpaexESLjcWQa4NerDECUiuYJ_a1IOrlL7oV1BuhU"

if __name__ == "__main__":

    # download from dropbox
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    dbx_df = dropbox_download(
        dbx, folder="", subfolder="", name="LinkedIn_labelled.csv"
    )
    lnkd_df = pd.read_csv(BytesIO(dbx_df))
    lnkd_df = lnkd_df[
        [   "id",
            "time",
            "event",
            "label",
        ]
    ]
    # rename and recode datetime variable
    lnkd_df["time"] = (lnkd_df["time"] - lnkd_df["time"].min()) 
    # minimum length to filter out sequences
    #minlen = 10
    lnkd_df["seqlen"] = lnkd_df.groupby(["id"])["time"].transform("count")
    #lnkd_df = lnkd_df[lnkd_df["seqlen"] >= minlen]
    # label encoding categorical variable
    lnkd_df["event"] = lnkd_df["event"].astype("category")
    lnkd_df["event"] = lnkd_df["event"].cat.codes
    # seqlen stats
    grouped_df = lnkd_df[["id", "seqlen"]]
    grouped_df = lnkd_df.groupby("id").agg({"seqlen": "mean"}).reset_index()
    grouped_df = grouped_df[["seqlen"]]
    print("seqlen stats")
    print("mean", grouped_df["seqlen"].mean())
    print(grouped_df.quantile([0.25, 0.5, 0.75]))
    lnkd_df.drop(columns=["seqlen"], inplace=True)
    # encoding label
    lnkd_df["label"] = lnkd_df["label"].astype("category")
    lnkd_df["label"] = lnkd_df["label"].cat.codes

    unique_ids = lnkd_df["id"].unique().tolist()
    gt_clusters = []

    # number of classes
    print("# event =", len(lnkd_df["event"].unique().tolist()))

    # number of labels
    print("# labels =", len(lnkd_df["label"].unique().tolist()))

    save_path = "data/Linkedin"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    i = 1
    for id0 in unique_ids:
        curr_df = lnkd_df[lnkd_df["id"] == id0].copy()
        gt_clusters.append(int(curr_df["label"].mean()))
        curr_df.drop(columns=["id", "label"], inplace=True)
        curr_df.reset_index(drop=True, inplace=True)
        curr_df.to_csv(os.path.join(save_path, str(i) + ".csv"))
        i += 1

    # saving gt cluster labels
    pd.DataFrame(gt_clusters, columns=["cluster_id"]).to_csv(
        os.path.join(save_path, "clusters.csv")
    )

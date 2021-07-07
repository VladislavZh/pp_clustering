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
        dbx, folder="", subfolder="", name="booking_challenge_tpp_labeled.csv"
    )
    bk_df = pd.read_csv(BytesIO(dbx_df))
    # local path to input file
    # bk_df = pd.read_csv("../data/booking_challenge_tpp_labeled.csv")
    bk_df = bk_df[
        [
            "user_id",
            "device_class",
            "city_id",
            "checkin",
            "diff_inout",
            "diff_checkin",
            "label",
        ]
    ]
    # rename and recode datetime variable
    bk_df.rename(columns={"checkin": "time"}, inplace=True)
    bk_df["time"] = pd.to_datetime(bk_df["time"])
    bk_df["time"] = (bk_df["time"] - bk_df["time"].min()) / np.timedelta64(1, "D")
    # minimum length to filter out sequences
    minlen = 10
    bk_df["seqlen"] = bk_df.groupby(["user_id"])["time"].transform("count")
    bk_df = bk_df[bk_df["seqlen"] >= minlen]
    # label encoding categorical variables
    cat_vars = ["device_class", "city_id", "diff_inout", "diff_checkin"]
    for catvar in cat_vars:
        # mapping = {k: i for i, k in enumerate(bk_df[catvar].unique())}
        # bk_df[catvar] = bk_df[catvar].map(mapping)
        bk_df[catvar] = bk_df[catvar].astype("category")
        bk_df[catvar] = bk_df[catvar].cat.codes
    # seqlen stats
    grouped_df = bk_df[["user_id", "seqlen"]]
    grouped_df = bk_df.groupby("user_id").agg({"seqlen": "mean"}).reset_index()
    grouped_df = grouped_df[["seqlen"]]
    print("seqlen stats")
    print("mean", grouped_df["seqlen"].mean())
    print(grouped_df.quantile([0.25, 0.5, 0.75]))
    bk_df.drop(columns=["seqlen"], inplace=True)
    # encoding label
    bk_df["label"] = bk_df["label"].astype("category")
    bk_df["label"] = bk_df["label"].cat.codes

    unique_ids = bk_df["user_id"].unique().tolist()
    gt_clusters = []

    # number of classes
    print("# device_class =", len(bk_df["device_class"].unique().tolist()))
    print("# city_id =", len(bk_df["city_id"].unique().tolist()))
    print("# diff_checkin =", len(bk_df["diff_checkin"].unique().tolist()))
    print("# diff_inout =", len(bk_df["diff_inout"].unique().tolist()))
    # number of labels
    print("# labels =", len(bk_df["label"].unique().tolist()))

    save_path = "data/booking"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    i = 1
    for id0 in unique_ids:
        curr_df = bk_df[bk_df["user_id"] == id0].copy()
        gt_clusters.append(int(curr_df["label"].mean()))
        curr_df.drop(columns=["user_id", "label"], inplace=True)
        curr_df.reset_index(drop=True, inplace=True)
        curr_df.to_csv(os.path.join(save_path, str(i) + ".csv"))
        i += 1

    # saving gt cluster labels
    pd.DataFrame(gt_clusters, columns=["cluster_id"]).to_csv(
        os.path.join(save_path, "clusters.csv")
    )

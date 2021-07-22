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
    atm_test = dropbox_download(
        dbx, folder="", subfolder="", name="ATM_test_day.csv"
    )
    test_df = pd.read_csv(BytesIO(atm_test))
    atm_train = dropbox_download(
        dbx, folder="", subfolder="", name="ATM_train_day.csv"
    )
    train_df = pd.read_csv(BytesIO(atm_train))
    # start splitting
    atm_df = pd.concat([test_df, train_df])
    atm_df = atm_df.sort_values(by=['id', 'time'])
    unique_ids = atm_df['id'].unique().tolist()
    
    gt_clusters = []
    save_path = "data/ATM"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    i = 1
    for id0 in unique_ids:
        curr_df = atm_df[atm_df["id"] == id0].copy()
        curr_df.drop(columns=["id"], inplace=True)
        curr_df.reset_index(drop=True, inplace=True)
        gt_clusters.append(curr_df.mode()["event"][0])
        curr_df.to_csv(os.path.join(save_path, str(i) + ".csv"))
        i += 1

    # saving gt cluster labels
    pd.DataFrame(gt_clusters, columns=["cluster_id"]).to_csv(
        os.path.join(save_path, "clusters.csv")
    )

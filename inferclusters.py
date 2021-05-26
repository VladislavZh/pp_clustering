import torch
import time
import numpy as np
from utils.metrics import purity
import math
from models.LSTM import LSTMMultiplePointProcesses
from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise
import pickle
import json
import os
import pandas as pd


if __name__ == "__main__":

    dataset = "sin_K5_C5"
    dataset = "Age"
    datapath = os.path.join("data", dataset)
    experpath = os.path.join("experiments", dataset)
    experpath = os.path.join(experpath, "exp_0")
    modelweights = os.path.join(experpath, "last_model.pt")
    with open(os.path.join(experpath, "args.json")) as json_file:
        config = json.load(json_file)
    n_steps = config["n_steps"]
    n_classes = config["n_classes"]
    data, target = get_dataset(datapath, n_classes, n_steps)

    model = LSTMMultiplePointProcesses(
        n_classes + 1,
        config["hidden_size"],
        config["num_layers"],
        n_classes,
        config["n_clusters"],
        n_steps,
        dropout=config["dropout"],
    ).to(config["device"])
    optimizer = torch.optim.Adam(
        model.parameters(), config["lr"], weight_decay=config["weight_decay"]
    )
    model = torch.load(modelweights)
    model.eval()

    trainer = TrainerClusterwise(
        model,
        optimizer,
        config["device"],
        data,
        config["n_clusters"],
        target=target,
        epsilon=config["epsilon"],
        max_epoch=config["max_epoch"],
        max_m_step_epoch=config["max_m_step_epoch"],
        lr=config["lr"],
        random_walking_max_epoch=config["random_walking_max_epoch"],
        true_clusters=config["true_clusters"],
        upper_bound_clusters=config["upper_bound_clusters"],
        lr_update_tol=config["lr_update_tol"],
        lr_update_param=config["lr_update_param"],
        min_lr=config["min_lr"],
        updated_lr=config["updated_lr"],
        batch_size=config["batch_size"],
        verbose=config["verbose"],
        best_model_path=None,
        max_computing_size=config["max_computing_size"],
        full_purity=config["full_purity"],
    )

    start_time = time.time()
    lambdas = trainer.model(trainer.X)
    trainer.gamma = trainer.compute_gamma(lambdas)
    print(trainer.X.shape)
    print(trainer.n_clusters)
    print(trainer.gamma.shape)
    clusters = torch.argmax(trainer.gamma, dim=0)
    print(clusters)
    print(clusters.shape)
    print(clusters.max())
    end_time = time.time()
    # save results
    res_df = pd.read_csv(os.path.join(datapath, "clusters.csv"))
    res_df["time"] = round(end_time - start_time,5)
    res_df["seqlength"] = 0
    for index, row in res_df.iterrows():
        seq_df = pd.read_csv(os.path.join(datapath, str(index + 1) + ".csv"))
        res_df.at[index, "seqlength"] = len(seq_df)
    
    #infclusters = {'inferred': clusters.cpu().numpy()}
    #res_df = pd.concat([res_df, pd.DataFrame(data=infclusters)])
    res_df['coh_cluster'] = clusters.cpu().numpy().tolist()
    savepath = os.path.join(experpath, "inferredclusters.csv")
    res_df.drop(res_df.columns[res_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    res_df.to_csv(savepath)

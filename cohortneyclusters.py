import argparse
import json
import os
import time

import pandas as pd
import torch

from models.LSTM import LSTMMultiplePointProcesses
from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="booking")
    parser.add_argument("--col_to_select", type=str, default=None)
    parser.add_argument("--experiment_n", type=str, default="exp_0")
    args = parser.parse_args()
    # path to dataset
    data_path = os.path.join("data", args.dataset)
    # path to experiment settings and weights
    tsfresh_path = os.path.join("experiments", args.dataset, "exp_0")
    exper_path = os.path.join("experiments", args.dataset, args.experiment_n)
    model_weights = os.path.join(exper_path, "last_model.pt")
    with open(os.path.join(exper_path, "args.json")) as json_file:
        config = json.load(json_file)
    n_steps = config["n_steps"]
    n_classes = config["n_classes"]
    #modeltype = "LSTM"
    # init model
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
    model = torch.load(model_weights, map_location=torch.device(config["device"]))
    model.modelname = "lstm"
    model.eval()
    # start
    start_time = time.time()
    data, target = get_dataset(
        data_path, model.num_classes, n_steps, args.col_to_select
    )

    trainer = TrainerClusterwise(
        model,
        optimizer,
        config["device"],
        data,
        model.num_clusters,
        exper_path=exper_path,
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

    # infer clusters
    if trainer.max_computing_size is None:
        lambdas = trainer.model(trainer.X.to(config["device"]))
        trainer.gamma = trainer.compute_gamma(lambdas)
        clusters = torch.argmax(trainer.gamma, dim=0)
    else:
        # large datasets
        trainer_data = trainer.X
        lenX = len(trainer.X)
        for i in range(0, lenX // trainer.max_computing_size + 1):
            trainer.X = trainer_data[
                i * trainer.max_computing_size : (i + 1) * trainer.max_computing_size
            ]
            lambdas = trainer.model(trainer.X.to(config["device"]))
            trainer.gamma = torch.zeros(trainer.n_clusters, len(trainer.X)).to(
                config["device"]
            )
            trainer.gamma = trainer.compute_gamma(lambdas)
            if i == 0:
                clusters = torch.argmax(trainer.gamma, dim=0)
            else:
                curr_clusters = torch.argmax(trainer.gamma, dim=0)
                clusters = torch.cat((clusters, curr_clusters), dim=0)

    end_time = time.time()
    # saving results
    res_df = pd.read_csv(os.path.join(tsfresh_path, "tsfresh_clusters.csv"))
    res_df["time"] = round(end_time - start_time, 5)
    res_df["coh_cluster"] = clusters.detach().cpu().numpy().tolist()
    savepath = os.path.join(exper_path, "compare_clusters.csv")
    res_df.drop(
        res_df.columns[res_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    res_df.to_csv(savepath)

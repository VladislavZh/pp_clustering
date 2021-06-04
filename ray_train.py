import json
import pickle
from functools import partial

import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from models.LSTM import LSTMMultiplePointProcesses
from utils.data_preprocessor import get_dataset
from utils.file_system_utils import create_folder
from utils.metrics import purity
from utils.trainers import TrainerClusterwise


def train_cohortney(args, data_dir=None):
    # reading datasets
    if args["verbose"]:
        print("Reading dataset")
    # data_dir = args['path_to_files']
    data, target = get_dataset(data_dir, args["n_classes"], args["n_steps"])
    if args["verbose"]:
        print("Dataset is loaded")

    # preparing folders
    if args["verbose"]:
        print("Preparing folders")
    create_folder("experiments")
    path = args["save_dir"].split("/")
    for i in range(len(path)):
        create_folder("experiments/" + "/".join(path[: i + 1]))
    path_to_results = "experiments/" + args["save_dir"]

    # iterations over runs
    i = 0
    all_results = []
    while i < args["n_runs"]:
        if args["verbose"]:
            print("Run {}/{}".format(i + 1, args["n_runs"]))
        model = LSTMMultiplePointProcesses(
            args["n_classes"] + 1,
            args["hidden_size"],
            args["num_layers"],
            args["n_classes"],
            args["n_clusters"],
            args["n_steps"],
            dropout=args["dropout"],
        ).to(args["device"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
        )
        best_model_path = path_to_results + "/exp_{}".format(i) + "/best_model.pt"
        create_folder(path_to_results + "/exp_{}".format(i))
        exp_folder = path_to_results + "/exp_{}".format(i)
        trainer = TrainerClusterwise(
            model,
            optimizer,
            args["device"],
            data,
            args["n_clusters"],
            target=target,
            epsilon=args["epsilon"],
            max_epoch=args["max_epoch"],
            max_m_step_epoch=args["max_m_step_epoch"],
            lr=args["lr"],
            random_walking_max_epoch=args["random_walking_max_epoch"],
            true_clusters=args["true_clusters"],
            upper_bound_clusters=args["upper_bound_clusters"],
            lr_update_tol=args["lr_update_tol"],
            lr_update_param=args["lr_update_param"],
            min_lr=args["min_lr"],
            updated_lr=args["updated_lr"],
            batch_size=args["batch_size"],
            verbose=args["verbose"],
            best_model_path=best_model_path if args["save_best_model"] else None,
            max_computing_size=args["max_computing_size"],
            full_purity=args["full_purity"],
        )
        losses, results, cluster_part, stats = trainer.train()

        # results check
        if cluster_part is None:
            if args["verbose"]:
                print("Solution failed")
            continue
        trainer.model.eval()
        if trainer.max_computing_size is None:
            lambdas = trainer.model(trainer.X.to(args["device"]))
            trainer.gamma = trainer.compute_gamma(lambdas)
            clusters = torch.argmax(trainer.gamma, dim=0)
        gtlabels_df = pd.read_csv(os.path.join(args["path_to_files"], "clusters.csv"))
        purity = purity(
            gtlabels_df["cluster_id"].to_list(), clusters.cpu().numpy().to_list()
        )
        tune.report(purity=purity)
        # saving results
        with open(exp_folder + "/args.json", "w") as f:
            json.dump(args, f)
        torch.save(trainer.model.state_dict(), exp_folder + "/last_model.pth")
        i += 1


if __name__ == "__main__":

    # reading parameters
    with open("base_config.json", "r") as f:
        base_params = json.load(f)

    base_params["n_dropout"] = tune.uniform(0.2, 0.7)
    #with open("ray_config.json", "r") as f:
    #    ray_params = json.load(f)

    scheduler = ASHAScheduler(
        max_t=base_params["max_epoch"], grace_period=base_params["max_epoch"]//10, reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=list(["n_dropout"]),
        metric_columns=["purity", "training_iteration"],
    )

    analysis = tune.run(
        partial(train_cohortney, data_dir="/pp_clustering/data/sin_K5_C5"),
        metric="purity",
        mode="max",
        num_samples=3,
        verbose=True,
        scheduler=scheduler,
        progress_reporter=reporter,
        config=base_params,
        resources_per_trial={"cpu": 6, "gpu": 1},
    )
    print("best config: ", analysis.get_best_config(metric="purity", mode="max"))

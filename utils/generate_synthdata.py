import json
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as spsp
import torch
from tick.base import TimeFunction
from tick.hawkes import (HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw,
                         HawkesKernelTimeFunc, SimuHawkes,
                         SimuHawkesExpKernels, SimuHawkesMulti,
                         SimuHawkesSumExpKernels)
from tick.plot import plot_point_process


class Random_sin:
    def __init__(self, C):
        self.b = np.zeros((C, C))
        self.w = np.zeros((C, C))
        self.s = np.zeros((C, C))
        for c in range(C):
            for c1 in range(C):
                self.b[c, c1], self.w[c, c1], self.s[c, c1] = (
                    random_me(),
                    random_me(),
                    random_me(),
                )

    def get_kernel(self, c, c1, t):
        return self.b[c, c1] * (1 - np.cos(self.w[c, c1] * (t - self.s[c, c1])))


class Random_sin_trunc:
    def __init__(self, C):
        self.b = np.zeros((C, C))
        self.w = np.zeros((C, C))
        self.s = np.zeros((C, C))
        for c in range(C):
            for c1 in range(C):
                self.b[c, c1], self.w[c, c1], self.s[c, c1] = (
                    random_me(),
                    random_me(),
                    random_me(),
                )

    def get_kernel(self, c, c1, t):
        return (
            2
            * self.b[c, c1]
            * np.round(
                self.b[c, c1]
                * (1 - np.cos(self.w[c, c1] * (t - self.s[c, c1])))
                / 2
                / self.b[c, c1]
            )
        )


def random_me():
    return np.pi / 5 + np.pi / 5 * np.random.random()


def random_seed(seed):
    """
    Fix seeds for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_seq_to_df(timestamps):
    """
    Format sequence as pandas dataframe
    """
    ts = []
    cs = []

    for c, tc in enumerate(timestamps):
        cs += [c] * len(tc)
        ts += list(tc)
    s = list(zip(ts, cs))
    s = list(sorted(s, key=lambda x: x[0]))
    s = np.array(s)
    df = pd.DataFrame(data=s, columns=["time", "event"])

    return df


def convert_clusters_to_dfs(clusters):
    dfs = []
    cluster_ids = []

    for cl_id, cluster in enumerate(clusters):
        cluster_ids += [cl_id] * len(cluster)
        for realiz in cluster:
            df = convert_seq_to_df(realiz)
            dfs.append(df)

    return dfs, cluster_ids


def simulate_hawkes(
    n_nodes,
    n_decays,
    n_realiz,
    end_time,
    dt,
    seed,
    max_jumps=1000,
    adj_density=0.25,
):
    tss = []

    baselines = np.random.rand(n_nodes) / n_nodes
    # baselines = spsp.rand(1, n_nodes, density=0.5).toarray()[0] / n_nodes
    decays = 5 * np.random.rand(n_nodes, n_nodes)
    adjacency = spsp.rand(n_nodes, n_nodes, density=adj_density).toarray()
    # Simulation

    for i in range(n_realiz):
        seed_ = seed + i if seed is not None else None
        hawkes = SimuHawkesExpKernels(
            baseline=baselines, decays=decays, adjacency=adjacency, seed=seed_
        )
        hawkes.adjust_spectral_radius(0.8)
        hawkes.max_jumps = max_jumps

        hawkes.end_time = end_time
        hawkes.verbose = False

        hawkes.track_intensity(dt)
        hawkes.simulate()
        tss.append(hawkes.timestamps)

        #plot_point_process(hawkes)

    return tss


def simulate_hawkes_sin(n_nodes, n_swcays, n_realiz, end_time, dt, seed, ax_jumps=1000):
    tss = []
    # The elements of exogenous base intensity are sampled uniformly from [0, 1]
    baseline = np.random.random(n_nodes)

    hawkes = SimuHawkes(baseline=baseline, seed=seed)

    body = Random_sin(n_nodes)
    times = np.linspace(0, 1, 10)  # 0, 1, 10, time = 10
    for c in range(n_nodes):
        for c1 in range(n_nodes):
            ys = body.get_kernel(c, c1, times)
            tf = TimeFunction([times, ys])
            kernel = HawkesKernelTimeFunc(tf)
            hawkes.set_kernel(c, c1, kernel)

    multi = SimuHawkesMulti(hawkes, n_simulations=n_realiz)
    multi.end_time = [end_time for i in range(n_realiz)]
    multi.simulate()

    tss = multi.timestamps
    # for i in range(n_realiz):
    #    plot_point_process(multi.get_single_simulation(i))

    return tss


def simulate_hawkes_sin_trunc(
    n_nodes, n_swcays, n_realiz, end_time, dt, seed, max_jumps=1000
):
    tss = []
    # The elements of exogenous base intensity are sampled uniformly from [0, 1]
    baseline = np.random.random(n_nodes)

    hawkes = SimuHawkes(baseline=baseline, force_simulation=True)

    body = Random_sin_trunc(n_nodes)
    times = np.linspace(0, 1, 10)
    for c in range(n_nodes):
        for c1 in range(n_nodes):
            ys = body.get_kernel(c, c1, times)
            tf = TimeFunction([times, ys])
            kernel = HawkesKernelTimeFunc(tf)
            hawkes.set_kernel(c, c1, kernel)

    multi = SimuHawkesMulti(hawkes, n_simulations=n_realiz)
    multi.end_time = [end_time for i in range(n_realiz)]
    multi.simulate()

    tss = multi.timestamps
    return tss


def simulate_clusters(
    n_clusters,
    n_nodes,
    n_decays,
    n_realiz,
    end_time,
    dt,
    seed,
    max_jumps,
    adj_density=0.25,
    sim_type="exp",
):
    """
    Switch between various types of data generating processes
    """
    clusters = []
    for i in range(n_clusters):
        # specific seed for each cluster
        seed_ = seed + i if seed is not None else None

        if sim_type == "exp":
            clusters.append(
                simulate_hawkes(
                    n_nodes,
                    n_decays,
                    n_realiz,
                    end_time,
                    dt,
                    seed_,
                    max_jumps,
                    adj_density,
                )
            )
        elif sim_type == "sin":
            clusters.append(
                simulate_hawkes_sin(
                    n_nodes,
                    n_decays,
                    n_realiz,
                    end_time,
                    dt,
                    seed_,
                    max_jumps,
                )
            )
        elif sim_type == "trunc":
            clusters.append(
                simulate_hawkes_sin_trunc(
                    n_nodes,
                    n_decays,
                    n_realiz,
                    end_time,
                    dt,
                    seed_,
                    max_jumps,
                )
            )
        else:
            raise ValueError("Unknown sim type")

    return clusters


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--n_clusters", type=int, required=True, help="number of clusters, required"
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        required=True,
        help="number of classes, required",
    )
    parser.add_argument(
        "--n_per_cluster",
        type=int,
        default=400,
        help="number of realizations per cluster, default is 400",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="random seed for reproducibility, default is 23",
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        required=True,
        help="type of data generating process",
    )
    args = parser.parse_args()

    save_dir = (
        "data/"
        + args.sim_type
        + "_K"
        + str(args.n_clusters)
        + "_C"
        + str(args.n_nodes)
    )

    print("Simulating...")
    random_seed(seed=args.seed)
    clusters = simulate_clusters(
        args.n_clusters,
        args.n_nodes,
        n_decays=3,
        n_realiz=args.n_per_cluster,
        end_time=100,
        dt=0.01,
        seed=args.seed,
        max_jumps=1000,
        adj_density=0.25,
        sim_type=args.sim_type,
    )
    dfs, cluster_ids = convert_clusters_to_dfs(clusters)
    print("Saving...")
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    for i, df in enumerate(dfs):
        df.to_csv(Path(save_dir, f"{i+1}.csv").open("w"))

    pd.DataFrame(data=np.array(cluster_ids), columns=["cluster_id"]).to_csv(
        Path(save_dir, f"clusters.csv").open("w")
    )
    # saving metadata
    info_json = {}
    info_json["seed"] = args.seed
    info_json["number of clusters"] = args.n_clusters
    info_json["number of classes"] = args.n_nodes
    info_json["length"] = args.n_per_cluster * args.n_clusters
    info_json["sim type"] = args.sim_type
    # to do
    info_json["log likelihood"] = 0
    with open(os.path.join(save_dir, "info.json"), "w") as infofile:
        json.dump(info_json, infofile)
    print("Finished")

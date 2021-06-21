import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("science")


# Fig 8
exper_path = "../../experiments/ablation_study"

starts = ["test_lr_", "test_n_clusters_", "test_n_steps_", "test_upper_bound_clusters_"]
x_ax_titles = [
    "Learning rate",
    "Initial number of clusters",
    "Number of steps in partitions",
    "Upper bound of the number of clusters",
]

k = 0
for st in starts:

    plot_means = []
    plot_stds = []
    x_labels = []

    for foldername in os.listdir(exper_path):
        # obtaining summary statistics
        if foldername.startswith(st):
            print(foldername.split("_"))
            x_labels.append(foldername.split("_")[-1])
            n_runs = len(os.listdir(os.path.join(exper_path, foldername)))
            purities = np.zeros(n_runs)
            for i in range(0, n_runs):
                res_file = os.path.join(
                    exper_path, foldername, "exp_" + str(i), "results.pkl"
                )
                with open(res_file, "rb") as f:
                    res_list = pickle.load(f)
                # leaving only last purity
                purities[i] = res_list[-1][1]

            plot_means.append(np.mean(purities))
            plot_stds.append(np.std(purities))

    # plotting
    x_labels = [float(x) for x in x_labels]
    sort_index = np.argsort(x_labels)
    print(x_labels)
    print(sort_index)
    x_labels = [x_labels[i] for i in sort_index] 
    print(x_labels)
    print(plot_means)
    plot_means = [plot_means[i] for i in sort_index] 
    print(plot_means)
    print(plot_stds)
    plot_stds = [plot_stds[i] for i in sort_index] 
    print(plot_stds)
    plt.errorbar(x_labels, plot_means, yerr=plot_stds)
    plt.xlabel(x_ax_titles[k])
    plt.ylabel("Purity")
    plt.savefig(st + "purity.pdf", dpi=400, bbox_inches="tight")
    plt.clf()
    k += 1

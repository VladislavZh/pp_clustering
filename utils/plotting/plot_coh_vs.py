import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import save_formatted


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


plt.style.use("science")


# Figure 9
dataset_names = ["new_sin_K5_C5", "K3_C5", "Age"]
coh_ll = []
for dataset in dataset_names:
    exper_path = os.path.join("../../experiments", dataset)
    # TODO: calculate true ll of dgp of dataset
    true_ll = 0
    n_runs = 10
    for i in range(0, n_runs):
        res_file = os.path.join(exper_path, "exp_" + str(i), "results.pkl")
        with open(res_file, "rb") as f:
            res_list = pickle.load(f)
        # leaving only log ll
        ll = [r[0] for r in res_list]
        if i == 0:
            total_ll = [ll]
        else:
            total_ll.append(ll)

    # reshape: from run index to epoch index
    total_ll = np.array([np.array(ll) for ll in total_ll])
    total_ll = total_ll.T
    mean_ll = np.mean(total_ll, axis=1)
    mean_ll = mean_ll - true_ll
    coh_ll.append(mean_ll)

data_coh = coh_ll
# TODO: data_zhu is to be added
data_zhu = [[3 * 10 ** 6], [3 * 10 ** 5], [2 * 10 ** 6]]
dataset_names = ["sin\_K5\_C5", "K3\_C5", "Age"]

with open("plot_config.json") as config:
    plot_settings = json.load(config)

with plt.style.context(plot_settings["style"]):
    fig, ax = plt.subplots()
    bpl = ax.boxplot(
        data_coh,
        positions=np.array(range(len(data_coh))) * 2.0 - 0.4,
        sym="",
        widths=0.6,
    )
    bpr = ax.boxplot(
        data_zhu,
        positions=np.array(range(len(data_zhu))) * 2.0 + 0.4,
        sym="",
        widths=0.6,
    )
    # colors are from http://colorbrewer2.org/
    set_box_color(bpl, "#D7191C")
    set_box_color(bpr, "#2C7BB6")

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c="#D7191C", label="Cohortney")
    ax.plot([], c="#2C7BB6", label="Zhu")
    ax.legend(fontsize="x-small")
    # label x-ticks
    ax.set_xticks(range(0, len(dataset_names) * 2, 2), dataset_names)
    ax.set_xlim(-2, len(dataset_names) * 2)

    # TODO: automate
    # draw gray strips
    ax.axvspan(-0.9, 0.9, facecolor="gray", alpha=0.5)
    ax.axvspan(3.1, 4.9, facecolor="gray", alpha=0.5)

save_formatted(fig, ax, plot_settings, "fig9.pdf", xlabel=None, ylabel=None, title=None)

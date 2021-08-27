import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import save_formatted

# Figure 11
exper_path = "../../experiments/new_sin_K5_C5"
with open("plot_config.json") as config:
    plot_settings = json.load(config)

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
# mean and std
mean_ll = np.mean(total_ll, axis=1)
std_ll = np.std(total_ll, axis=1)
epochs = list(range(1, len(res_list) + 1))

with plt.style.context(plot_settings["style"]):
    fig, ax = plt.subplots()
    plt.errorbar(epochs[3:], mean_ll[3:], yerr=std_ll[3:], label="Cohortney")
    plt.legend(loc="upper right")

data_name = exper_path.split("/")[-1]
save_formatted(
    fig,
    ax,
    plot_settings,
    save_path=data_name + "_negll.pdf",
    xlabel="Epochs",
    ylabel=None,
    title=None,
)

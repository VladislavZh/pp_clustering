import json
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib import ticker

from plotting_utils import save_formatted

# Figure 10
exper_path = "../../experiments/Age/exp_0"
res_file = os.path.join(exper_path, "results.pkl")

with open(res_file, "rb") as f:
    res_list = pickle.load(f)

epochs = list(range(1, len(res_list) + 1))
# negative log likelihood
ll = [r[0] for r in res_list]
# purities
pur = [r[1] for r in res_list]
# cluster partition
clust_part = [r[2] for r in res_list]
# number of clusters
n_clust = [r[3] for r in res_list]
# time
time_fr_st = [r[4] for r in res_list]

metrics = [ll, pur, clust_part, n_clust, time_fr_st]
metrics_name = [
    "Negative\nLog Likelihood",
    "Purity",
    "Cluster\npartition",
    "Number\nof clusters",
    "Time",
]
data_name = exper_path.split("/")[-2]
with open("plot_config.json") as config:
    plot_settings = json.load(config)

for i, m in enumerate(metrics):

    with plt.style.context(plot_settings["style"]):
        fig, axs = plt.subplots()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axs.yaxis.set_major_formatter(formatter)
        axs.plot(epochs, metrics[i])
    save_formatted(
        fig,
        axs,
        plot_settings,
        save_path=data_name
        + "_"
        + metrics_name[i].replace("\n", " ")
        + "_learncurve.pdf",
        xlabel="Epoch",
        ylabel=None,
        title=metrics_name[i],
    )

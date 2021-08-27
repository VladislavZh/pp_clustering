import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from plotting_utils import save_formatted

dfolders = ["../../data/ATM", "../../data/Linkedin"]
plot_array = []


# process dataset to obtain log-intervals
for df in dfolders:
    listfiles = os.listdir(df)
    if "clusters.csv" in listfiles:
        listfiles.remove("clusters.csv")
    if "info.json" in listfiles:
        listfiles.remove("info.json")
    print(df)

    tmp_array = []
    for f in listfiles:
        curr_f = pd.read_csv(os.path.join(df, f))
        curr_f = curr_f.sort_values(by=["time"])
        # first diff
        curr_f = curr_f[["time"]]
        curr_f = curr_f.diff()
        # remove first row with nan
        curr_f = curr_f[1:]
        delta = curr_f["time"].values
        delta = delta[delta > 0]
        log_delta = list(np.log(delta))
        tmp_array.extend(log_delta)

    tmp_array.sort()
    plot_array.append(tmp_array)


# plot hist for Fig 3
plt.style.use("science")
with open("plot_config.json") as config:
    plot_settings = json.load(config)

for i in range(len(plot_array)):

    data_name = dfolders[i].split("/")[-1]
    currdata = plot_array[i]

    with plt.style.context(plot_settings["style"]):
        fig, ax = plt.subplots()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        plt.hist(currdata, bins=20)

    save_formatted(
        fig,
        ax,
        plot_settings,
        save_path=data_name + "_hist.pdf",
        xlabel="Log inter-event time",
        ylabel="Count",
        title=data_name,
    )

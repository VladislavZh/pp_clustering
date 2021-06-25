import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dfolders = ["../../data/ATM", "../../data/IPTV", "../../data/Linkedin"]
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

for i in range(len(plot_array)):

    data_name = dfolders[i].split("/")[-1]
    currdata = plot_array[i]

    # q25, q75 = np.percentile(currdata,[.25,.75])
    # bin_width = 2*(q75 - q25)*len(currdata)**(-1/3)
    # bins = round((currdata[-1] - currdata[0])/bin_width)
    # print("Freedman–Diaconis number of bins:", bins)
    plt.hist(currdata, bins=20)
    plt.ylabel("Count")
    plt.xlabel("Log inter-event time")
    plt.title(data_name)
    plt.savefig(data_name + "_hist.pdf", dpi=400)
    plt.clf()

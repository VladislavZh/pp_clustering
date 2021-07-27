import numpy as np
import pandas as pd
from calc_purity import cohortney_tsfresh_stats

ROUND_DEC = 2

outside_results = dict()
outside_results["DMHP"] = {
    "exp_K2_C5": "0.91 0.00",
    "exp_K3_C5": "0.66 0.00",
    "exp_K4_C5": "0.80 0.08",
    "exp_K5_C5": "0.58 0.03",
    "sin_K2_C5": "0.98 0.05",
    "sin_K3_C5": "0.98 0.00",
    "sin_K4_C5": "0.58 0.06",
    "sin_K5_C5": "0.75 0.05",
    "trunc_K2_C5": "1.00 0.00",
    "trunc_K3_C5": "0.67 0.00",
    "trunc_K4_C5": "0.99 0.00",
    "trunc_K5_C5": "0.88 0.09",
    "IPTV": "0.34 0.03",
    "Age": "0.38 0.01",
    "Linkedin": "0.31 0.01",
    "ATM": "0.64 0.02",
    "Booking": "-",
}

outside_results["Soft DTW"] = {
    "exp_K2_C5": "0.50 0.00",
    "exp_K3_C5": "0.33 0.00",
    "exp_K4_C5": "0.25 0.00",
    "exp_K5_C5": "-",
    "sin_K2_C5": "0.50 0.00",
    "sin_K3_C5": "0.33 0.00",
    "sin_K4_C5": "0.25 0.00",
    "sin_K5_C5": "0.20 0.00",
    "trunc_K2_C5": "0.50 0.00",
    "trunc_K3_C5": "0.33 0.00",
    "trunc_K4_C5": "0.25 0.00",
    "trunc_K5_C5": "0.20 0.00",
    "IPTV": "0.32 0.00",
    "Age": "-",
    "Linkedin": "0.20 0.00",
    "ATM": "0.14 0.00",
    "Booking": "0.33 0.00",
}

outside_results["K-shape"] = {
    "exp_K2_C5": "0.50 0.00",
    "exp_K3_C5": "0.33 0.00",
    "exp_K4_C5": "0.25 0.00",
    "exp_K5_C5": "0.20 0.00",
    "sin_K2_C5": "0.50 0.00",
    "sin_K3_C5": "0.33 0.00",
    "sin_K4_C5": "0.25 0.00",
    "sin_K5_C5": "0.20 0.00",
    "trunc_K2_C5": "0.50 0.00",
    "trunc_K3_C5": "0.33 0.00",
    "trunc_K4_C5": "0.25 0.00",
    "trunc_K5_C5": "0.20 0.00",
    "IPTV": "0.32 0.00",
    "Age": "-",
    "Linkedin": "0.20 0.00",
    "ATM": "0.14 0.00",
    "Booking": "0.33 0.00",
}

outside_results["K-means ps"] = {
    "exp_K2_C5": "0.89 0.00",
    "exp_K3_C5": "0.52 0.00",
    "exp_K4_C5": "0.60 0.00",
    "exp_K5_C5": "0.58 0.00",
    "sin_K2_C5": "0.93 0.00",
    "sin_K3_C5": "0.85 0.00",
    "sin_K4_C5": "0.51 0.00",
    "sin_K5_C5": "0.56 0.00",
    "trunc_K2_C5": "1.00 0.00",
    "trunc_K3_C5": "0.45 0.00",
    "trunc_K4_C5": "0.75 0.00",
    "trunc_K5_C5": "0.44 0.00",
    "IPTV": "0.34 0.00",
    "Age": "0.35 0.00",
    "Linkedin": "0.20 0.00",
    "ATM": "-",
    "Booking": "-",
}

outside_results["DMHP_time"] = {
    "exp_K2_C5": "3044",
    "exp_K3_C5": "49313",
    "exp_K4_C5": "122645",
    "exp_K5_C5": "219504",
    "sin_K2_C5": "71492",
    "sin_K3_C5": "160122",
    "sin_K4_C5": "236643",
    "sin_K5_C5": "234628",
    "trunc_K2_C5": "1540",
    "trunc_K3_C5": "40850",
    "trunc_K4_C5": "77167",
    "trunc_K5_C5": "141554",
    "IPTV": "74135",
    "Age": "55327",
    "Linkedin": "57404",
    "-": "-",
}


def format_for_table2(arr: np.array) -> str:
    """
    Formats summary statistics of np.array as "mean +- std"
    """
    cell = (
        str(round(np.mean(arr), ROUND_DEC)) + " " + str(round(np.std(arr), ROUND_DEC))
    )

    return cell


if __name__ == "__main__":

    datasets = ["exp_K2_C5", "exp_K3_C5", "exp_K4_C5", "exp_K5_C5"]
    methods = ["cohortney", "gmm", "kmeans"]
    # print table 2
    print("Printing Table 2...")
    cols = [
        "Dataset",
        "COHORTNEY",
        "DMHP",
        "Soft",
        "K-",
        "K-means",
        "K-means0",
        "GMM",
    ]
    cols = ["\textbf{" + c + "}" for c in cols]
    table2 = pd.DataFrame(columns=cols)
    seccols = [
        "",
        "(ours)",
        "[45]",
        "DTW",
        "Shape",
        "partitions",
        "tsfresh",
        "tsfresh",
    ]
    seccols = ["\textbf{" + c + "}" for c in seccols]
    table2.loc[0] = seccols
    nr_wins = [0] * (len(cols) - 1)
    for i in range(0, len(datasets)):
        print("Formatting results of dataset", datasets[i])
        res_dict = cohortney_tsfresh_stats(datasets[i], methods)
        coh = np.array(res_dict["cohortney"]["purities"])
        coh_cell = format_for_table2(coh)

        kmeans_ts = np.array(res_dict["kmeans"]["purities"])
        gmm_ts = np.array(res_dict["gmm"]["purities"])
        kmeansts_cell = str(round(np.mean(kmeans_ts), ROUND_DEC))
        gmmts_cell = str(round(np.mean(gmm_ts), ROUND_DEC))

        symbolic = [
            datasets[i].replace("_", "\_"),
            coh_cell,
            outside_results["DMHP"][datasets[i]],
            outside_results["Soft DTW"][datasets[i]],
            outside_results["K-shape"][datasets[i]],
            outside_results["K-means ps"][datasets[i]],
            kmeansts_cell,
            gmmts_cell,
        ]
        # finding max and 2nd max
        numeric = [
            float(symbolic[i].split(" ")[0]) if symbolic[i] != "-" else 0.0
            for i in range(1, len(symbolic))
        ]
        first = second = -1
        for j in range(0, len(numeric)):
            if first < numeric[j]:
                second = first
                first = numeric[j]
            elif second < numeric[j] and first != numeric[j]:
                second = numeric[j]

        for j in range(0, len(numeric)):
            if numeric[j] == first:
                nr_wins[j] += 1
                # make max bold
                symbolic[j + 1] = "\textbf{" + symbolic[j + 1] + "}"
            elif numeric[j] == second:
                # make second max underlined
                symbolic[j + 1] = "\\underline{" + symbolic[j + 1] + "}"
        # add plus-minus
        symbolic = [symbolic[0]] + [
            symbolic[i].replace(" ", "$\pm$") for i in range(1, len(symbolic))
        ]
        table2.loc[i + 1] = symbolic
    
    maxnum = max(nr_wins)
    for i in range(len(nr_wins)):
        if nr_wins[i] == maxnum:
            nr_wins[i] = "\textbf{" + str(nr_wins[i]) + "}"

    table2.loc[i + 2] = ["Nr. of wins"] + nr_wins

    table2.to_latex(
        buf="table2.tex", index=False, escape=False, column_format="lccccccc"
    )
    print("Finished")
    
    # print table 3
    print("Printing Table 3...")
    ldatasets = [
        "exp_K2_C5",
        "exp_K3_C5",
        "exp_K4_C5",
        "exp_K5_C5",
        "sin_K2_C5",
        "sin_K3_C5",
        "sin_K4_C5",
        "sin_K5_C5",
    ]
    rdatasets = [
        "trunc_K2_C5",
        "trunc_K3_C5",
        "trunc_K4_C5",
        "trunc_K5_C5",
        "IPTV",
        "Age",
        "Linkedin",
        "-",
    ]
    methods = ["cohortney"]
    assert len(ldatasets) == len(rdatasets), "error: table is not balanced"
    cols = ["Dataset", "COHORTNEY", "DMHP", "Dataset0", "COHORTNEY0", "DMHP0"]
    cols = ["\textbf{" + c + "}" for c in cols]
    table3 = pd.DataFrame(
        columns = cols
    )

    for i in range(0, len(ldatasets)):
        print("Formatting results of dataset", ldatasets[i])
        if ldatasets[i] != "-":
            res_dict = cohortney_tsfresh_stats(ldatasets[i], methods)
            coh_l = res_dict["cohortney"]["train_time"] / res_dict["n_runs"]
            # coh_l = str(round(coh_l, ROUND_DEC))
            coh_l = str(int(coh_l))
        else:
            coh_l = "-"
        print("Formatting results of dataset", rdatasets[i])
        if rdatasets[i] != "-":
            res_dict = cohortney_tsfresh_stats(rdatasets[i], methods)
            coh_r = res_dict["cohortney"]["train_time"] / res_dict["n_runs"]
            # coh_r = str(round(coh_r, ROUND_DEC))
            coh_r = str(int(coh_r))
        else:
            coh_r = "-"
        table3.loc[i] = [
            ldatasets[i].replace('_','\_'),
            coh_l,
            outside_results["DMHP_time"][ldatasets[i]],
            rdatasets[i].replace('_','\_'),
            coh_r,
            outside_results["DMHP_time"][rdatasets[i]],
        ]

    table3.to_latex(buf="table3.tex", index=False, escape=False, column_format="cccccc")
    print("Finished")

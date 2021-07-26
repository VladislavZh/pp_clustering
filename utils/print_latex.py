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
        str(round(np.mean(arr), ROUND_DEC))
        + "+-"
        + str(round(np.std(arr), ROUND_DEC))
        + "$"
    )

    return cell


if __name__ == "__main__":

    datasets = ["exp_K2_C5", "exp_K3_C5"]
    methods = ["cohortney", "gmm", "kmeans"]
    # print table 2
    print("Printing Table 2...")
    table2 = pd.DataFrame(
        columns=[
            "\0333[1m" + "Dataset" + "\033[0m",
            "COHORTNEY",
            "DMHP",
            "Soft",
            "K-",
            "K-means",
            "K-means",
            "GMM",
        ]
    )
    table2.loc[0] = [
        "",
        "(ours)",
        "[45]",
        "DTW",
        "Shape",
        "partitions",
        "tsfresh",
        "tsfresh",
    ]
    for i in range(0, len(datasets)):
        print("Formatting results of dataset", datasets[i])
        res_dict = cohortney_tsfresh_stats(datasets[i], methods)
        coh = np.array(res_dict["cohortney"]["purities"])
        kmeans_ts = np.array(res_dict["kmeans"]["purities"])
        gmm_ts = np.array(res_dict["gmm"]["purities"])
        coh_cell = format_for_table2(coh)
        kmeansts_cell = format_for_table2(kmeans_ts)
        gmmts_cell = format_for_table2(gmm_ts)
        table2.loc[i+1] = [
            datasets[i],
            coh_cell,
            outside_results["DMHP"][datasets[i]],
            outside_results["Soft DTW"][datasets[i]],
            outside_results["K-shape"][datasets[i]],
            outside_results["K-means ps"][datasets[i]],
            kmeansts_cell,
            gmmts_cell,
        ]

    table2.to_latex(buf="table2.tex", index=False, column_format="lccc")
    print("Finished")
    # print table 3
    print("Printing Table 3...")
    ldatasets = ["exp_K2_C5", "exp_K3_C5", "exp_K4_C5", "exp_K5_C5", "sin_K2_C5", "sin_K3_C5", "sin_K4_C5", "sin_K5_C5"]
    rdatasets = ["trunc_K2_C5", "trunc_K3_C5", "trunc_K4_C5", "trunc_K5_C5", "IPTV", "Age", "Linkedin", "-"]
    methods = ["cohortney"]
    assert len(ldatasets) == len(rdatasets), "error: table is not balanced"
    table3 = pd.DataFrame(
        columns=["Dataset", "COHORTNEY", "DMHP", "Dataset0", "COHORTNEY0", "DMHP0"]
    )

    for i in range(0, len(ldatasets)):
        print("Formatting results of dataset", ldatasets[i])
        if ldatasets[i] != "-":
            res_dict = cohortney_tsfresh_stats(ldatasets[i], methods)
            print(res_dict["n_runs"])
            coh_l = res_dict["cohortney"]["train_time"] / res_dict["n_runs"]
            coh_l = str(round(coh_l, ROUND_DEC))
        else:
            coh_l = "-" 
        print("Formatting results of dataset", rdatasets[i])
        if rdatasets[i] != "-":
            res_dict = cohortney_tsfresh_stats(rdatasets[i], methods)
            coh_r = res_dict["cohortney"]["train_time"] / res_dict["n_runs"]
            coh_r = str(round(coh_r, ROUND_DEC))
        else:
            coh_r = "-"
        table3.loc[i] = [
            ldatasets[i],
            coh_l,
            outside_results["DMHP_time"][ldatasets[i]],
            rdatasets[i],
            coh_r,
            outside_results["DMHP_time"][rdatasets[i]],
        ]

    table3.to_latex(buf="table3.tex", index=False, column_format="cccccc")
    print("Finished")

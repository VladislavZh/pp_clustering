from typing import List

import argparse
import numpy as np
import pandas as pd
import sys
from get_metrics import cohortney_tsfresh_stats


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
    "ATM": "0",
    "-": "-",
}


def format_row_table2(symbolic: List[str], nr_wins: List[int]) -> List[str]:
    """
    Make max value bold and second max value underlined,
    adjust nr_wins row accordingly
    """
    # finding max and 2nd max
    numeric = [
        float(symbolic[j].split(" ")[0]) if symbolic[j] != "-" else 0.0
        for j in range(1, len(symbolic))
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
        symbolic[j].replace(" ", "$\pm$") for j in range(1, len(symbolic))
    ]

    return symbolic, nr_wins


def format_for_table2(arr: np.array) -> str:
    """
    Formats summary statistics of np.array as "mean +- std"
    """
    cell = (
        "{:.2f}".format(round(np.mean(arr), ROUND_DEC))
        + " "
        + "{:.2f}".format(round(np.std(arr), ROUND_DEC))
    )

    return cell


if __name__ == "__main__":

    datasets_cat = {
        "Exp": ["exp_K2_C5", "exp_K3_C5", "exp_K4_C5", "exp_K5_C5"],
        "Sin": ["sin_K2_C5", "sin_K3_C5", "sin_K4_C5", "sin_K5_C5"],
        "Trunc": ["trunc_K2_C5", "trunc_K3_C5", "trunc_K4_C5", "trunc_K5_C5"],
        "Real": ["Age", "IPTV", "Linkedin", "ATM"],
    }
    methods = ["cohortney", "gmm", "kmeans"]
    # print table 2
    available_metrics = [ "purities", "adj_mut_info_score", "adj_rand_score", "v_meas_score", "f_m_score"]
    parser = argparse.ArgumentParser(description='settings for table 2')
    parser.add_argument("--table2_metric", default="purities")
    args = parser.parse_args() 
    table2_metric = args.table2_metric
    if table2_metric not in available_metrics:
        print("please choose available metrics, they are:")
        print(available_metrics)
        sys.exit(1)

    print("Printing Table 2 with", table2_metric)
    cols = [
        "Dataset",
        "COHORTNEY",
        "DMHP",
        "K-shape",
        "K-means",
        "K-means0",
        "GMM",
    ]
    dolanmore_res = pd.DataFrame(columns=cols)
    dolanmore_res_sum = pd.DataFrame(columns=cols)
    cols = ["\textbf{" + c + "}" for c in cols]
    table2 = pd.DataFrame(columns=cols)
    table2_sum = pd.DataFrame(columns=cols)
    seccols = [
        "",
        "(ours)",
        "[45]",
        "(soft DTW)",
        "partitions",
        "tsfresh",
        "tsfresh",
    ]
    seccols = ["\textbf{" + c + "}" for c in seccols]
    table2.loc[0] = seccols
    table2_sum.loc[0] = seccols
    nr_wins = [0] * (len(cols) - 1)
    nr_wins_sum = [0] * (len(cols) - 1)

    table2_index = 0
    table2_sum_index = 0
    for ds_type, ds_list in datasets_cat.items():
        table2_sum_index += 1
        print("Formatting results of type", ds_type)
        coh_total = []
        dmhp_total = []
        kmeansts_total = []
        gmmts_total = []
        kshape_total = []
        kmeansps_total = []

        for ds in ds_list:
            table2_index += 1
            print("Formatting results of dataset", ds)
            res_dict = cohortney_tsfresh_stats(ds, methods)
            coh = np.array(res_dict["cohortney"][table2_metric])
            coh_total.extend(coh)
            coh_cell = format_for_table2(coh)
            dmhp = np.array(res_dict["zhu"][table2_metric])
            dmhp_total.extend(dmhp)
            dmhp_cell = format_for_table2(dmhp)
            kmeansts = np.array(res_dict["kmeans"][table2_metric])
            kmeansts_total.extend(kmeansts)
            kmeansts_cell = "{:.2f}".format(round(np.mean(kmeansts), ROUND_DEC))
            gmmts = np.array(res_dict["gmm"][table2_metric])
            gmmts_total.extend(gmmts)
            gmmts_cell = "{:.2f}".format(round(np.mean(gmmts), ROUND_DEC))
            kshape = np.array(res_dict["k_shape"][table2_metric])
            kshape_total.extend(kshape)
            kshape_cell = format_for_table2(kshape)
            #kshape_cell = "{:.2f}".format(round(np.mean(kshape), ROUND_DEC))
            kmeansps = np.array(res_dict["k_means_softdtw"][table2_metric])
            kmeansps_total.extend(kmeansps)
            kmeansps_cell = format_for_table2(kmeansps)
            #kmeansps_cell = "{:.2f}".format(round(np.mean(kmeansps), ROUND_DEC))

            symbolic = [
                ds.replace("_", "\_"),
                coh_cell,
                dmhp_cell,
                kshape_cell,
                kmeansps_cell,
                kmeansts_cell,
                gmmts_cell,
            ]
            dm_row = [
                ds, 
                float(coh_cell.split(" ")[0]) if coh_cell != "-" else 0.0,
                float(dmhp_cell.split(" ")[0]) if dmhp_cell != "-" else 0.0,
                float(kshape_cell.split(" ")[0]) if kshape_cell != "-" else 0.0,
                float(kmeansps_cell.split(" ")[0]) if kmeansps_cell != "-" else 0.0,
                float(kmeansts_cell),
                float(gmmts_cell),
            ]
            symbolic, nr_wins = format_row_table2(symbolic, nr_wins)
            table2.loc[table2_index] = symbolic
            dolanmore_res.loc[table2_index-1] = dm_row
            last_row = table2_index

        # summarizing
        coh_cell = format_for_table2(coh_total)
        dmhp_cell = format_for_table2(dmhp_total)
        kshape_cell = format_for_table2(kshape_total)
        kmeansps_cell = format_for_table2(kmeansps_total)
        kmeansts_cell = "{:.2f}".format(round(np.mean(kmeansts_total), ROUND_DEC))
        gmmts_cell = "{:.2f}".format(round(np.mean(gmmts_total), ROUND_DEC))
        #kshape_cell = "{:.2f}".format(round(np.mean(kshape_total), ROUND_DEC))
        #kmeansps_cell = "{:.2f}".format(round(np.mean(kmeansps_total), ROUND_DEC))
        symbolic = [
            ds_type,
            coh_cell,
            dmhp_cell,
            kshape_cell,
            kmeansps_cell,
            kmeansts_cell,
            gmmts_cell,
        ]
        dm_row = [
            ds_type, 
            float(coh_cell.split(" ")[0]) if coh_cell != "-" else 0.0,
            float(dmhp_cell.split(" ")[0]) if coh_cell != "-" else 0.0,
            float(kshape_cell.split(" ")[0]) if kshape_cell != "-" else 0.0,
            float(kmeansps_cell.split(" ")[0]) if kmeansps_cell != "-" else 0.0,
            float(kmeansts_cell),
            float(gmmts_cell),
        ]
        symbolic, nr_wins_sum = format_row_table2(symbolic, nr_wins_sum)
        table2_sum.loc[table2_sum_index] = symbolic
        dolanmore_res_sum.loc[table2_index-1] = dm_row
        last_row_sum = table2_sum_index

    maxnum = max(nr_wins)
    for j in range(len(nr_wins)):
        if nr_wins[j] == maxnum:
            nr_wins[j] = "\textbf{" + str(nr_wins[j]) + "}"
    maxnum = max(nr_wins_sum)
    for j in range(len(nr_wins_sum)):
        if nr_wins_sum[j] == maxnum:
            nr_wins_sum[j] = "\textbf{" + str(nr_wins_sum[j]) + "}"

    table2.loc[last_row + 1] = ["Nr. of wins"] + nr_wins
    table2_sum.loc[last_row_sum + 1] = ["Nr. of wins"] + nr_wins_sum

    table2.to_latex(
        buf=table2_metric + "_table2.tex",
        index=False,
        escape=False,
        column_format="lccccccc",
    )
    table2_sum.to_latex(
        buf=table2_metric + "_sum_table2.tex",
        index=False,
        escape=False,
        column_format="lccccccc",
    )
    dolanmore_res.to_csv("utils/plotting/"+ table2_metric + "_dm_res.csv")
    dolanmore_res_sum.to_csv("utils/plotting/" + table2_metric + "_sum_dm_res.csv")
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
        "ATM",
    ]
    methods = ["cohortney"]
    assert len(ldatasets) == len(rdatasets), "error: table is not balanced"
    cols = ["Dataset", "COHORTNEY", "DMHP", "Dataset0", "COHORTNEY0", "DMHP0"]
    cols = ["\textbf{" + c + "}" for c in cols]
    table3 = pd.DataFrame(columns=cols)
    cols = ["Dataset", "COHORTNEY", "DMHP"]
    cols = ["\textbf{" + c + "}" for c in cols]
    table3_sum = pd.DataFrame(columns=cols)

    print("Formatting summarized version...")
    i = 0
    for ds_type, ds_list in datasets_cat.items():
        total_time_coh = 0
        total_time_dmhp = 0
        total_runs = 0
        for ds in ds_list:
            res_dict = cohortney_tsfresh_stats(ds, methods)
            total_time_coh += res_dict["cohortney"]["train_time"]
            total_runs += res_dict["n_runs"]
            total_time_dmhp += int(outside_results["DMHP_time"][ds])
        table3_sum.loc[i] = [
            ds_type,
            str(int(total_time_coh / total_runs)),
            str(int(total_time_dmhp / len(ds_list))),
        ]
        i += 1

    table3_sum.to_latex(
        buf="sum_table3.tex", index=False, escape=False, column_format="lcc"
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
            ldatasets[i].replace("_", "\_"),
            coh_l,
            outside_results["DMHP_time"][ldatasets[i]],
            rdatasets[i].replace("_", "\_"),
            coh_r,
            outside_results["DMHP_time"][rdatasets[i]],
        ]

    table3.to_latex(buf="table3.tex", index=False, escape=False, column_format="cccccc")
    print("Finished")


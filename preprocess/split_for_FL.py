import os
import pandas as pd
import numpy as np
from main_step2 import split_traintest
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rawdata_path", type=str, default="RAWDATA_PATH")
    parser.add_argument("--inputdata_path", type=str, default="INPUTDATA_PATH")
    return parser


def main():
    args = get_parser().parse_args()
    src_list = ["mimic3", "eicu", "mimic4"]
    label_list = ["los3", "los7", "dx", "mort", "readm", "im_disch", "fi_ac"]
    target_list = ["input_ids", "dpe_ids", "type_ids"]

    client_list = [
        "mimic3_mv",
        "mimic3_cv",
        "eicu_73",
        "eicu_264",
        "eicu_420",
        "eicu_243",
        "eicu_458",
        "eicu_338",
        "eicu_443",
    ]
    # No need split mimic4
    id_dict = {
        "pid": {client: [] for client in client_list},
        "index": {client: [] for client in client_list},
    }

    # MIMIC-III split

    fold_mimic3 = pd.read_csv(
        os.path.join(args.inputdata_path, "mimic3", "fold", "fold_100.csv")
    )
    icu_mimic3 = pd.read_csv(os.path.join(args.rawdata_path, "mimic3", "ICUSTAYS.csv"))
    icu_mimic3.rename(columns={"HADM_ID": "pid"}, inplace=True)
    icu_mimic3 = icu_mimic3[icu_mimic3["pid"].isin(fold_mimic3["pid"].values)]
    icu_mimic3 = icu_mimic3.groupby("pid").first().reset_index()

    fold_mimic3 = fold_mimic3.merge(
        icu_mimic3[["pid", "DBSOURCE"]], how="left", on="pid"
    )

    id_dict["pid"]["mimic3_mv"] = fold_mimic3[fold_mimic3["DBSOURCE"] == "metavision"][
        "pid"
    ].tolist()
    id_dict["pid"]["mimic3_cv"] = fold_mimic3[fold_mimic3["DBSOURCE"] == "carevue"][
        "pid"
    ].tolist()
    id_dict["index"]["mimic3_mv"] = fold_mimic3[
        fold_mimic3["DBSOURCE"] == "metavision"
    ].index.tolist()
    id_dict["index"]["mimic3_cv"] = fold_mimic3[
        fold_mimic3["DBSOURCE"] == "carevue"
    ].index.tolist()

    # eicu split
    eicu_pat = pd.read_csv(os.path.join(args.rawdata_path, "eicu", "patient.csv"))
    eicu_pat.rename(columns={"patientunitstayid": "pid"}, inplace=True)

    fold_eicu = pd.read_csv(
        os.path.join(args.inputdata_path, "eicu", "fold", "fold_100.csv")
    )
    eicu_pat = eicu_pat[eicu_pat["pid"].isin(fold_eicu["pid"].values)]

    fold_eicu = fold_eicu.merge(eicu_pat[["pid", "hospitalid"]], how="left", on="pid")

    hos = []
    for hos_id, value in zip(
        fold_eicu["hospitalid"].value_counts().index,
        fold_eicu["hospitalid"].value_counts(),
    ):
        hos.append(hos_id)

    hos_top7 = hos[:7]
    print("top 7 hospital in eicu", hos_top7)

    for hos_id in hos_top7:
        hos_index = fold_eicu[fold_eicu["hospitalid"] == hos_id].index.tolist()
        hos_pid = fold_eicu[fold_eicu["hospitalid"] == hos_id]["pid"].tolist()
        id_dict["index"]["eicu_" + str(hos_id)] = hos_index
        id_dict["pid"]["eicu_" + str(hos_id)] = hos_pid

    # npy, labe, fold split for FL clients
    for client in id_dict["pid"].keys():
        src = client.split("_")[0]
        client_src_path = os.path.join(args.inputdata_path, client)
        src_path = os.path.join(args.inputdata_path, src)
        for data in target_list:
            victim = np.load(
                os.path.join(src_path, "npy", data + ".npy"), allow_pickle=True
            )
            victim = victim[id_dict["index"][client]]
            np.save(os.path.join(client_src_path, "npy", data + ".npy"), victim)
        for label in label_list:
            label_npy = np.load(
                os.path.join(src_path, "label", label + ".npy"), allow_pickle=True
            )
            label_npy = label_npy[id_dict["index"][client]]
            np.save(os.path.join(client_src_path, "label", label + ".npy"), label_npy)
        fold = pd.read_csv(os.path.join(src_path, "fold", "fold_100.csv"))
        fold = fold.loc[id_dict["index"][client]].reset_index(drop=True)
        fold = split_traintest(fold, check=True)
        fold.to_csv(
            os.path.join(client_src_path, "fold", label + "fold_100.csv"), index=False
        )


if __name__ == "__main__":
    main()

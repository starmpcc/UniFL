import argparse
import json
import os
import random
import time
import warnings

import more_itertools as mit
import numpy as np
import pandas as pd
from fold_split import random_split, stratified_split
from preprocess_utils import *
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def complete_dataframe(src, table_names, config, save_path):
    table_names = [i.split("/")[-1] for i in table_names]
    lab = pd.read_pickle(
        os.path.join(save_path, src, "descemb_whole", f"{table_names[0]}.pkl")
    )
    pre = pd.read_pickle(
        os.path.join(save_path, src, "descemb_whole", f"{table_names[1]}.pkl")
    )
    inp = pd.read_pickle(
        os.path.join(save_path, src, "descemb_whole", f"{table_names[2]}.pkl")
    )
    df = pd.concat([inp, lab, pre], axis=0)
    print(
        ">> Concat inputevents, lab, prescriptions: ",
        df.shape,
        inp.shape[0] + lab.shape[0] + pre.shape[0],
    )

    sorted_df = df.sort_values(["ORDER"], ascending=True).reset_index(drop=True)
    df_agg = sorted_df.groupby(["ID"]).agg(lambda x: x.tolist())
    print(">> Sort by time and liniearize events for each HADM_ID")

    # Read cohort dataframe including label info.
    cohort_df = pd.read_pickle(os.path.join(save_path, f"{src}_cohort.pkl"))
    cohort_df = cohort_df[
        [
            config["ID"][src],
            "readmission",
            "mortality",
            "los_3day",
            "los_7day",
            "final_acuity",
            "imminent_discharge",
            "diagnosis",
        ]
    ]
    cohort_df.rename(columns={config["ID"][src]: "ID"}, inplace=True)
    print(">> Read cohort dataframe to generate labels")

    # Merge concat dataframe & cohort dataframe
    pred_df = pd.merge(df_agg, cohort_df, how="left", on="ID")

    label = {
        "ID": "pid",
        "mortality": "mort",
        "los_3day": "los3",
        "los_7day": "los7",
        "readmission": "readm",
        "diagnosis": "dx",
        "final_acuity": "fi_ac",
        "imminent_discharge": "im_disch",
    }

    label_df = pred_df.rename(columns=label)

    label_df.to_csv(os.path.join(save_path, src, "label_df.csv"))

    return label_df


def split_traintest(label_df, check):

    # One-hot multi-label for diagnosis label
    label_df["dx"] = label_df["dx"].apply(lambda x: list(map(int, x)))
    mlb = MultiLabelBinarizer()
    multihot = mlb.fit_transform(label_df["dx"]).tolist()
    label_df["dx"] = pd.Series(multihot)

    # Make random seeds
    seed_list = [2020, 2021, 2022, 2023, 2024]
    test_split = 10
    val_split = 9

    df = label_df.copy()
    for seed in seed_list:
        print("seed split start : ", seed)
        df = stratified_split(df, seed, test_split, val_split)
        df = random_split(df, seed, test_split, val_split)

    # Check
    if check:
        pred_tasks = ["mort", "los3", "los7", "readm", "dx", "im_disch", "fi_ac"]

        # for stratified split
        print("stratified split sanity check !\n")
        for col in pred_tasks:
            for seed in seed_list:
                if col in df.columns:
                    print(
                        "train = 1/ valid =2 / test= 0 / remove= -1 \n",
                        df[col + f"_{seed}_strat"].value_counts(),
                    )
                    print(
                        "train label ratio \n ",
                        df.groupby([col, col + f"_{seed}_strat"]).size()
                        / len(df)
                        * 100,
                    )

        # for random split
        print("random split sanity check !\n")
        print(
            "train = 1/ valid =2 / test= 0 / remove= -1 \n",
            df[f"{seed}_rand"].value_counts(),
        )
        for col in pred_tasks:
            if col in df.columns:
                print(
                    "train label ratio \n ",
                    df.groupby([col, f"{seed}_rand"]).size() / len(df) * 100,
                )

    return df


def dataframe2numpy(df, root, src):

    np.save(os.path.join(root, src, "inputs.npy"), df.event_token.to_numpy())
    np.save(os.path.join(root, src, "types.npy"), df.type_token.to_numpy())
    np.save(os.path.join(root, src, "dpes.npy"), df.dpe_token.to_numpy())

    pred_tasks = ["mort", "los3", "los7", "readm", "dx"]
    for task in pred_tasks:
        np.save(os.path.join(root, src, "label", f"{task}.npy"), df[task].to_numpy())


def count_events_per_8192(types, max_len):

    event_count = []

    for type_ids in types:

        # Flatten data
        type_ids = sum(type_ids, [])

        # The first token should be [CLS]
        type_ids.insert(0, 1)

        # Truncate
        if len(type_ids) > max_len - 1:
            flatten_types = np.array(type_ids[:max_len]).astype(np.int32)

            timetoken_mask = flatten_types == 4
            timetoken_idx = np.where(timetoken_mask)[0]
            split_idx = timetoken_idx.max()

            if split_idx == max_len - 1:
                split_idx = sorted(timetoken_idx)[-2]

            # The last token shold be [SEP]
            flatten_types[split_idx + 1] = 2

            # Fill blank with [PAD]
            flatten_types[split_idx + 2 :] = 0

        # Pad
        else:
            flatten_types = np.zeros(max_len).astype(np.int32)
            flatten_types[: len(type_ids)] = type_ids

            # The last token shold be [SEP]
            flatten_types[len(type_ids)] = 2

        # Append
        event_num = (flatten_types == 4).sum()
        event_count.append(event_num)

    return event_count


def match_eventlen(event, event_len, event_max_len, max_len):

    # 1. Match numbers between flat data and hierarchical data
    hi_event = event[:event_len]

    # 2. If the number of events exceeds the maximum, the event is discarded
    if len(hi_event) > event_max_len:
        return None
    else:
        return hi_event


def make_hi_data(inputs, types, dpes, event_count, event_max_len, word_max_len):

    sample_num = len(event_count) - (np.array(event_count) > event_max_len).sum()

    input_events_list = np.empty((sample_num, event_max_len, word_max_len)).astype(
        np.int16
    )
    type_events_list = np.empty((sample_num, event_max_len, word_max_len)).astype(
        np.int16
    )
    dpe_events_list = np.empty((sample_num, event_max_len, word_max_len)).astype(
        np.int16
    )
    iter = 0

    events_exceeded = 0

    for input_event, type_event, dpe_event, event_len in zip(
        inputs, types, dpes, event_count
    ):

        hi_input_event = match_eventlen(
            input_event, event_len, event_max_len, word_max_len
        )
        hi_type_event = match_eventlen(
            type_event, event_len, event_max_len, word_max_len
        )
        hi_dpe_event = match_eventlen(dpe_event, event_len, event_max_len, word_max_len)

        if hi_input_event is None:
            events_exceeded += 1
            continue

        input_words_list = []
        type_words_list = []
        dpe_words_list = []

        for input_ids, type_ids, dpe_ids in zip(
            hi_input_event, hi_type_event, hi_dpe_event
        ):

            # The first token should be [CLS]
            input_ids.insert(0, 101)
            type_ids.insert(0, 1)
            dpe_ids.insert(0, 0)

            if len(type_ids) > word_max_len - 1:

                time_token = input_ids[-1]
                hi_input_ids = np.array(input_ids[:word_max_len]).astype(np.int16)
                hi_type_ids = np.array(type_ids[:word_max_len]).astype(np.int16)
                hi_dpe_ids = np.array(dpe_ids[:word_max_len]).astype(np.int16)

                content_mask = np.where(hi_type_ids == 7)[0]
                content_indices = [
                    list(group) for group in mit.consecutive_groups(content_mask)
                ]
                # content_indices = [[100,101],[107,108]]

                if len(content_indices) == 0:
                    raise AssertionError("Edge case.. Need to check")

                split_indices = content_indices[-1][-1]

                if split_indices >= word_max_len - 2:
                    split_indices = content_indices[-2][-1]

                # The second last token shold be [TIME]
                hi_input_ids[split_indices + 1] = time_token
                hi_type_ids[split_indices + 1] = 4
                hi_dpe_ids[split_indices + 1] = 1

                # The last token shold be [SEP]
                hi_input_ids[split_indices + 2] = 102
                hi_type_ids[split_indices + 2] = 2
                hi_dpe_ids[split_indices + 2] = 0

                # Fill blank with [PAD]
                hi_input_ids[split_indices + 3 :] = 0
                hi_type_ids[split_indices + 3 :] = 0
                hi_dpe_ids[split_indices + 3 :] = 0

            else:
                hi_input_ids = np.zeros(word_max_len).astype(np.int16)
                hi_type_ids = np.zeros(word_max_len).astype(np.int16)
                hi_dpe_ids = np.zeros(word_max_len).astype(np.int16)

                hi_input_ids[: len(input_ids)] = input_ids
                hi_type_ids[: len(type_ids)] = type_ids
                hi_dpe_ids[: len(dpe_ids)] = dpe_ids

                # The last token shold be [SEP]
                hi_input_ids[len(input_ids)] = 102
                hi_type_ids[len(type_ids)] = 2
                hi_dpe_ids[len(dpe_ids)] = 0

            input_words_list.append(hi_input_ids)
            type_words_list.append(hi_type_ids)
            dpe_words_list.append(hi_dpe_ids)

        # Pad events to be 256 events
        empty_event = [
            [0] * word_max_len for i in range(event_max_len - len(input_words_list))
        ]
        input_words_list += empty_event
        type_words_list += empty_event
        dpe_words_list += empty_event

        input_events_list[iter] = input_words_list
        type_events_list[iter] = type_words_list
        dpe_events_list[iter] = dpe_words_list
        iter += 1

    print(f"{events_exceeded} events are discarded..")
    if iter != sample_num:
        raise AssertionError("Should be the same!")

    return {
        "inputs": input_events_list,
        "types": type_events_list,
        "dpes": dpe_events_list,
    }


def flatten_hi_data(hi_output, max_len):

    input_ids_list = []
    type_ids_list = []
    dpe_ids_list = []
    event_count = []

    for input_ids, type_ids, dpe_ids in zip(
        hi_output["inputs"], hi_output["types"], hi_output["dpes"]
    ):

        # Flatten data
        input_ids = input_ids.flatten()
        type_ids = type_ids.flatten()
        dpe_ids = dpe_ids.flatten()

        # Remove [PAD], [CLS], [SEQ]
        mask = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)

        input_ids = input_ids[~mask]
        type_ids = type_ids[~mask]
        dpe_ids = dpe_ids[~mask]

        padded_input_ids = np.zeros(max_len).astype(np.int16)
        padded_type_ids = np.zeros(max_len).astype(np.int16)
        padded_dpe_ids = np.zeros(max_len).astype(np.int16)

        # The first token shold be [CLS]
        padded_input_ids[0] = 101
        padded_type_ids[0] = 1
        padded_dpe_ids[0] = 0

        padded_input_ids[1 : len(input_ids) + 1] = input_ids
        padded_type_ids[1 : len(type_ids) + 1] = type_ids
        padded_dpe_ids[1 : len(dpe_ids) + 1] = dpe_ids

        # The last token shold be [SEP]
        padded_input_ids[len(input_ids) + 1] = 102
        padded_type_ids[len(type_ids) + 1] = 2
        padded_dpe_ids[len(dpe_ids) + 1] = 0

        input_ids_list.append(padded_input_ids)
        type_ids_list.append(padded_type_ids)
        dpe_ids_list.append(padded_dpe_ids)

        event_num = (padded_type_ids == 4).sum()
        event_count.append(event_num)

    return {
        "inputs": np.array(input_ids_list),
        "types": np.array(type_ids_list),
        "dpes": np.array(dpe_ids_list),
        "event_count": event_count,
    }


def final_check(flat_output, hi_output, word_max_len):

    print("FLAT OUTPUT CHECK")
    p_idx = random.randint(0, len(flat_output))

    for ii, ti, di in zip(
        flat_output["inputs"][p_idx][:128],
        flat_output["types"][p_idx][:128],
        flat_output["dpes"][p_idx][:128],
    ):
        print(ti, "\t", di, "\t", tokenizer.decode(ii))

    print("\n...\n")

    for ii, ti, di in zip(
        flat_output["inputs"][p_idx][-128:],
        flat_output["types"][p_idx][-128:],
        flat_output["dpes"][p_idx][-128:],
    ):
        print(ti, "\t", di, "\t", tokenizer.decode(ii))

    flat_event_num = (flat_output["types"][p_idx] == 4).sum()

    print("\nHIERARHCICAL OUTPUT CHECK")

    hi_event_num = 0
    for input_ids in hi_output["inputs"][p_idx]:
        if (input_ids == 0).sum() != word_max_len:
            hi_event_num += 1

    w_idx = 0
    for ii, ti, di in zip(
        hi_output["inputs"][p_idx][w_idx],
        hi_output["types"][p_idx][w_idx],
        hi_output["dpes"][p_idx][w_idx],
    ):
        print(ti, "\t", di, "\t", tokenizer.decode(ii))

    if flat_event_num != hi_event_num:
        raise AssertionError("Should be the same")
    return


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    return parser


def main():
    args = get_parser().parse_args()

    # Argument
    save = True
    config_path = "./json/config.json"

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    for src in ["mimic3", "eicu", "mimic4"]:
        os.makedirs(os.path.join(args.save_path, src), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, src, "label"), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, src, "fold"), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, src, "npy"), exist_ok=True)

        table_names = [elem["table_name"] for elem in config["Table"][src]]

        # [1] Find label and make fold splits
        if not os.path.isfile(os.path.join(args.save_path, "label_df.csv")):
            df = complete_dataframe(src, table_names, config, args.save_path)
        else:
            start = time.time()
            df = pd.read_csv(os.path.join(args.save_path, "label_df.csv"))
            end = time.time()
            print(">> Loading {} csv is done... {} [sec]".format(df.shape, end - start))

        if not os.path.isfile(os.path.join(args.save_path, "dpes.npy")):
            # [2] Convert df to numpy
            dataframe2numpy(df, args.save_path, src)

        start = time.time()
        inputs = np.load(
            os.path.join(args.save_path, src, "inputs.npy"), allow_pickle=True
        )
        types = np.load(
            os.path.join(args.save_path, src, "types.npy"), allow_pickle=True
        )
        dpes = np.load(os.path.join(args.save_path, src, "dpes.npy"), allow_pickle=True)
        end = time.time()
        print(
            ">> Loading numpy {} is done... {} [sec]".format(inputs.shape, end - start)
        )

        # [3] Fix-length data
        event_count = count_events_per_8192(types, max_len=8192)

        start = time.time()
        hi_output = make_hi_data(
            inputs, types, dpes, event_count, event_max_len=256, word_max_len=128
        )
        print(
            ">> Hierarhical data {} is done... {} [sec]".format(
                hi_output["inputs"].shape, time.time() - start
            )
        )

        start = time.time()
        flat_output = flatten_hi_data(hi_output, max_len=8192)
        print(
            ">> Flat data {} is done... {} [sec]".format(
                flat_output["inputs"].shape, time.time() - start
            )
        )

        mask = np.array(event_count) > 256
        print(f">> {mask.sum()} events are discared ..")

        if save:
            np.save(
                os.path.join(args.save_path, src, "npy", "input_ids.npy"),
                hi_output["inputs"],
            )
            np.save(
                os.path.join(args.save_path, src, "npy", "type_ids.npy"),
                hi_output["types"],
            )
            np.save(
                os.path.join(args.save_path, src, "npy", "dpe_ids.npy"),
                hi_output["dpes"],
            )

            pred_tasks = ["mort", "los3", "los7", "readm", "dx"]

            os.makedirs(os.path.join(args.save_path, src, "label"), exist_ok=True)
            for task in pred_tasks:
                pred_data = df[task].to_numpy()
                np.save(
                    os.path.join(args.save_path, src, "label", f"{task}.npy"),
                    pred_data[~mask],
                )
                print(f"[{task}] \t", pred_data[~mask].shape)

            split_traintest(df[~mask].reset_index(drop=False), check=False).to_csv(
                os.path.join(args.save_path, src, "fold", f"fold_100.csv")
            )


if __name__ == "__main__":
    main()

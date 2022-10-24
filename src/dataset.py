import os
import logging

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class HierarchicalEHRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        input_path,
        split,
        ratio,
        pred_target,
        seed,
        debug,
        max_word_len=128,
        max_seq_len=512,
        **kwargs,
    ):

        self.base_path = os.path.join(input_path, data)
        self.seed = seed
        self.split = split
        self.seed = seed

        self.debug = debug
        self.time_bucket_idcs = [
            idx for idx in range(4, 24)
        ]  # start range + bucket num + 1

        self.data_dir = os.path.join(self.base_path, "npy")
        self.fold_file = os.path.join(
            self.base_path, "fold", "fold_{}.csv".format(ratio)
        )

        self.pred_target = pred_target

        self.tokenizer = None

        label = np.load(
            os.path.join(self.base_path, "label", pred_target + ".npy"),
            allow_pickle=True,
        )
        if self.pred_target == "dx":
            mlb = MultiLabelBinarizer(
                classes=[str(i) for i in range(1, 19)],
            )
            label = mlb.fit_transform(label)

        self.label = torch.tensor(label, dtype=torch.long)

        self.hit_idcs = self.get_fold_indices()
        self.label = self.label[self.hit_idcs]

        logger.info(f"loaded {data} {len(self.hit_idcs)} {self.split} samples")
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len

        self.cls = 101

    def get_fold_indices(self, return_all=False):
        if self.split == "train":
            hit = 1
        elif self.split == "valid":
            hit = 2
        elif self.split == "test":
            hit = 0

        df = pd.read_csv(self.fold_file)
        if return_all:
            return np.arange(len(df))

        col_name = self.pred_target + "_" + str(2020) + "_strat"

        splits = df[col_name].values
        idcs = np.where(splits == hit)[0]

        return idcs

    def __len__(self):
        if self.debug:
            return len(self.hit_idcs) // 100
        return len(self.hit_idcs)

    def crop_to_max_size(self, arr, target_size):
        """
        arr: 1d np.array of indices
        """
        size = len(arr)
        diff = size - target_size
        if diff <= 0:
            return arr

        return arr[:target_size]

    def collator(self, samples):
        samples = [s for s in samples if s["input_ids"] is not None]
        if len(samples) == 0:
            return {}

        input = dict()
        out = dict()

        input["input_ids"] = [s["input_ids"] for s in samples]
        input["type_ids"] = [s["type_ids"] for s in samples]
        input["dpe_ids"] = [s["dpe_ids"] for s in samples]

        seq_sizes = []
        word_sizes = []
        for s in input["input_ids"]:
            seq_sizes.append(len(s))
            for w in s:
                word_sizes.append(len(w))

        target_seq_size = min(max(seq_sizes), self.max_seq_len)
        target_word_size = min(max(word_sizes), self.max_word_len)

        collated_input = dict()
        for k in input.keys():
            collated_input[k] = torch.zeros(
                (
                    len(input["input_ids"]),
                    target_seq_size,
                    target_word_size,
                )
            ).long()

        for i, seq_size in enumerate(seq_sizes):
            for j in range(len(input["input_ids"][i])):
                word_size = len(input["input_ids"][i][j])
                diff = word_size - target_word_size
                for k in input.keys():
                    if diff == 0:
                        pass
                    elif diff < 0:
                        try:
                            input[k][i][j] = np.append(input[k][i][j], [0] * -diff)
                        except ValueError:
                            input[k][i] = list(input[k][i])
                            input[k][i][j] = np.append(input[k][i][j], [0] * -diff)
                    else:
                        input[k][i][j] = np.array(input[k][i][j][: self.max_word_len])

            diff = seq_size - target_seq_size
            for k in input.keys():
                if k == "input_ids":
                    prefix = self.cls
                else:
                    prefix = 1
                input[k][i] = np.array(list(input[k][i]))
                if diff == 0:
                    collated_input[k][i] = torch.from_numpy(input[k][i])
                elif diff < 0:
                    padding = np.zeros(
                        (
                            -diff,
                            target_word_size - 1,
                        )
                    )
                    padding = np.concatenate(
                        [np.full((-diff, 1), fill_value=prefix), padding], axis=1
                    )
                    collated_input[k][i] = torch.from_numpy(
                        np.concatenate([input[k][i], padding], axis=0)
                    )
                else:
                    collated_input[k][i] = torch.from_numpy(
                        self.crop_to_max_size(input[k][i], target_seq_size)
                    )

        out["net_input"] = collated_input
        if "label" in samples[0]:
            out["label"] = torch.stack([s["label"] for s in samples])

        return out

    def __getitem__(self, index):
        fname = str(self.hit_idcs[index]) + ".npy"

        input_ids = np.load(
            os.path.join(self.data_dir, "input_ids", fname), allow_pickle=True
        )
        type_ids = np.load(
            os.path.join(self.data_dir, "type_ids", fname), allow_pickle=True
        )
        dpe_ids = np.load(
            os.path.join(self.data_dir, "dpe_ids", fname), allow_pickle=True
        )
        label = self.label[index]

        out = {
            "input_ids": input_ids,
            "type_ids": type_ids,
            "dpe_ids": dpe_ids,
            "label": label,
        }

        return out

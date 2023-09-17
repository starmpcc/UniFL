import argparse
import os

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    return parser


def main():
    args = get_parser().parse_args()

    src_list = [
        "mimic3_mv",
        "mimic3_cv",
        "mimic4",
        "eicu_73",
        "eicu_243",
        "eicu_264",
        "eicu_338",
        "eicu_420",
        "eicu_443",
        "eicu_458",
    ]

    data_list = ["input_ids", "type_ids", "dpe_ids"]

    for src in src_list:
        for data in data_list:
            data_dir = os.path.join(args.save_path, src, "npy", data)
            os.makedirs(data_dir, exist_ok=True)

            print("processing: " + data_dir)
            arr = np.load(
                os.path.join(args.save_path, src, "npy", f"{data}.npy"),
                allow_pickle=True,
            )
            for i, el in enumerate(arr):
                el = np.array(el)
                np.save(os.path.join(data_dir, f"{i}.npy"), el)


if __name__ == "__main__":
    main()

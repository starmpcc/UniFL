import numpy as np
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rawdata_path", type=str, default="RAWDATA_PATH")
    parser.add_argument("--inputdata_path", type=str, default="INPUTDATA_PATH")
    return parser


def main():
    args = get_parser().parse_args()

    srcs = ["mimic3", "eicu", "mimic4"]

    for src in srcs:
        input_path = os.path.join(args.inputdata_path, "npy", src)
        inputs = np.load(
            os.path.join(input_path, f"input_ids.npy"),
            allow_pickle=True,
        ).astype(np.int64)

        types = np.load(
            os.path.join(input_path, f"type_ids.npy"),
            allow_pickle=True,
        ).astype(np.int64)

        dpes = np.load(
            os.path.join(input_path, f"dpe_ids.npy"),
            allow_pickle=True,
        ).astype(np.int64)

        for idx, inp in enumerate(inputs):
            remove_zero_events = np.array([i for i in inp if np.any(i)])
            trunc_inp = np.array([(i[i != 0]).tolist() for i in remove_zero_events])
            len_events = len(remove_zero_events)
            len_tokens = [len(i) for i in trunc_inp]

            trunc_type = [
                i[: len_tokens[idx_i]].tolist()
                for idx_i, i in enumerate(types[idx][:len_events])
            ]
            trunc_dpes = [
                i[: len_tokens[idx_i]].tolist()
                for idx_i, i in enumerate(dpes[idx][:len_events])
            ]
            # breakpoint()

        np.save(
            os.path.join(input_path, "input_ids", f"{idx}.npy"),
            trunc_inp,
        )
        np.save(
            os.path.join(input_path, "type_ids", f"{idx}.npy"),
            trunc_type,
        )
        np.save(
            os.path.join(input_path, "dpe_ids", f"{idx}.npy"),
            trunc_dpes,
        )

    print("[{}] Finished creating npy datasets for {}...".format(src))


if __name__ == "__main__":
    main()

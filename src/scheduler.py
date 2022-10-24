import subprocess
import argparse
import os

datasets = [
    "mimic3_mv",
    "mimic3_cv",
    "mimic4",
    "eicu_73",
    "eicu_264",
    "eicu_420",
    "eicu_243",
    "eicu_458",
    "eicu_338",
    "eicu_443",
]
tasks = ["mort", "los3", "los7", "readm", "dx"]
seeds = ["2020", "2021", "2022", "2023", "2024"]
algorithms = ["fedavg", "fedbn", "fedprox", "fedpxn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=str, default="0,1", required=True
    )  # Assume A6000x2
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["single", "federated", "pooled"],
        required=True,
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument("--save_dir", type=str, requied=True)
    parser.add_argument("--eventencoder", type=str, default="rnn")

    args = parser.parse_args()

    exps_list = []
    if args.train_type == "single":
        for task in tasks:
            for dataset in datasets:
                for seed in seeds:
                    exps_list.append((task, [dataset], seed, "None"))
    elif args.train_type == "pooled":
        for task in tasks:
            for seed in seeds:
                exps_list.append((task, datasets, seed, "None"))
    else:
        for algorithm in algorithms:
            for task in tasks:
                for seed in seeds:
                    exps_list.append((task, datasets, seed, algorithm))
    runnings = [None] * (len(args.gpus.split(",")) // 2)

    exps_list.reverse()
    env = os.environ.copy()
    while exps_list:
        for i, run in enumerate(runnings):
            if (run is None) or (run.poll() is not None):
                exp = exps_list.pop()
                # Assume using 2 GPUs with DDP
                gpu = ",".join(args.gpus.split(",")[i * 2 : i * 2 + 2])
                print(f"Run Experiment {exp} on device {gpu}")
                runnings[i] = subprocess.Popen(
                    [
                        f"/bin/sh",
                        "-c",
                        f"python src/main.py --device_num {gpu} --train_type {args.train_type} --pred_target {exp[0]} --src_data {' '.join(exp[1])}\
                                --algorithm {exp[3]} --seed {exp[2]} --input_path {args.input_path} --save_dir {args.save_dir} --port {12355+i}",
                    ],
                    env=env,
                )
                if not exps_list:
                    break


if __name__ == "__main__":
    main()

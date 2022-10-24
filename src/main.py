import argparse
import logging
import os
import sys
import datetime

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from .federated import federated
from .base import base

os.environ["OMP_NUM_THREADS"] = "8"


available_srcs = [
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="0")

    # checkpoint configs
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, default="checkpoint")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # dataset
    parser.add_argument(
        "--train_type",
        choices=["single", "federated", "pooled"],
        type=str,
        default="single",
    )

    parser.add_argument(
        "--src_data",
        nargs="*",
        action="store",
        choices=available_srcs,
        type=str,
        default=available_srcs,
    )

    parser.add_argument("--ratio", choices=["0", "10", "100"], type=str, default="100")

    parser.add_argument(
        "--pred_target",
        choices=["mort", "los3", "los7", "readm", "dx"],
        type=str,
        default="mort",
        help="",
    )

    # trainer
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--valid_subsets", type=list, default=["valid", "test"])
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--apply_mean", action="store_true", default=None)

    # model hyper-parameter configs
    parser.add_argument(
        "--eventencoder", type=str, choices=["transformer", "rnn"], default="rnn"
    )
    parser.add_argument("--pred_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--type_token", action="store_true")
    parser.add_argument("--dpe", action="store_true")
    parser.add_argument("--pos_enc", action="store_true")
    parser.add_argument("--pred_pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--map_layers", type=int, default=1)
    parser.add_argument("--max_word_len", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--port", type=str, default="12357")
    parser.add_argument("--debug", action="store_true", default=False)

    # Hyper-parameters for Fedeated learning
    parser.add_argument(
        "--algorithm", type=str, default="None"
    )  # fedprox,fedbn,fedpxn,fedavg
    parser.add_argument("--communications", type=int, default=1000)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.01)  # FedProx, FedPxN

    # Wandb
    parser.add_argument("--wandb_entity_name", type=str, required=True)
    parser.add_argument("--wandb_project_name", type=str, required=True)

    # For Kubernetis Resume
    # Note that this function is only supported for FL
    parser.add_argument("--resume_dir", type=str, default=None)
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    print(args.src_data)

    # Resume experiment, assume that use whole available srcs and on federated learning
    current_exp = [
        args.train_type,
        args.pred_target,
        args.algorithm,
        str(args.seed),
        args.eventencoder,
    ]
    os.makedirs(args.save_dir, exist_ok=True)
    args.resume = False
    if args.resume_dir is not None:
        os.makedirs(args.resume_dir, exist_ok=True)
        prev_exps = os.listdir(args.resume_dir)

        for i in prev_exps:
            prev_conf = i.split("_")[2:7]
            if current_exp == prev_conf:
                args.resume = True
                args.exp_name = i
                logger.info(f"Resume checkpoint from {i}")
                break

    if not args.resume:
        args.exp_name = "_".join(
            [
                datetime.datetime.today().strftime("%m%d_%H%M"),
                args.train_type,
                args.pred_target,
                args.algorithm,
                str(args.seed),
                args.eventencoder,
                "_".join(args.src_data),
            ]
        )
        os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    if args.train_type == "federated":
        federated(args)
    elif args.train_type in ["single", "pooled"]:
        base(args)
    else:
        raise NotImplementedError()

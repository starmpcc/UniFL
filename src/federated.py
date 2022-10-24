import torch, os, random, requests
import torch.multiprocessing as mp
import numpy as np
import logging
import traceback

from .utils import trainer_utils as utils
from .utils import distributed_utils
from torch.utils.data import DataLoader
from .loss import PredLoss
from .dataset import HierarchicalEHRDataset
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from .model import UniHPF
from .metric import PredMetric

from .communication_func import communication
from .local_training import (
    train_naive,
    train_fedprox,
    inference,
)
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb
import torch.nn as nn
import torch
import signal
import types
from shutil import rmtree

logger = logging.getLogger(__name__)


def sigterm_handler(s: signal.Signals, f: types.FrameType) -> None:
    raise KeyboardInterrupt


def federated(args):
    signal.signal(signal.SIGTERM, sigterm_handler)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)

    args.device_ids = list(range(len(args.device_num.split(","))))
    print("device_number : ", args.device_ids)
    args.world_size = len(args.device_ids)

    # seed pivotting
    mp.set_sharing_strategy("file_system")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    trainer = FedTrainer(args)
    try:
        trainer.train()
        logger.info("done train")

    except KeyboardInterrupt:
        if trainer.is_data_parallel_master:
            save_dir = (
                os.path.join(
                    trainer.args.resume_dir,
                    trainer.args.exp_name,
                )
                + "/"
            )
            os.makedirs(save_dir, exist_ok=True)
            for c, data_name in enumerate(trainer.train_data):
                torch.save(
                    trainer.old_local[c].state_dict(),
                    save_dir + trainer.args.save_prefix + f"last_{data_name}.pt",
                )
            torch.save(
                trainer.old_server.state_dict(),
                save_dir + trainer.args.save_prefix + f"last_server.pt",
            )
            torch.save(
                trainer.old_metric,
                save_dir + trainer.args.save_prefix + f"last_metric.pt",
            )
            torch.save(
                trainer.old_early,
                save_dir + trainer.args.save_prefix + f"last_early.pt",
            )
            logger.info("Finish saving before interrupt")

            try:
                api_host = os.environ["NSML_RUN_METADATA_API"]
                api_secret = os.environ["NSML_RUN_SECRET"]
                requests.put(
                    f"{api_host}/v1/rescheduled",
                    headers={"X-NSML-Run-Secret": api_secret},
                    json={"rescheduled": True},
                ).raise_for_status()
            except:
                # Sometimes, the HTTP request might fail, but the training process should not be stopped.
                traceback.print_exc()


class FedTrainer:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix
        self.train_data = args.src_data

        self.datasets = dict()
        self.early_stopping_dict = dict()
        data_types = ["train"] + self.args.valid_subsets

        for split in data_types:
            self.datasets[split] = OrderedDict()
            for data in self.train_data:
                self.datasets[split][data] = self.load_dataset(split, data)
        client_weights = [len(i) for i in self.datasets["train"].values()]
        tot_len = sum(client_weights)
        self.client_weights = [i / tot_len for i in client_weights]

        self.server_model = UniHPF(args)
        self.local_models = [
            deepcopy(self.server_model) for _ in range(len(args.src_data))
        ]
        self.metric = PredMetric(self.args)
        self.param_name = [name for name, _ in self.server_model.named_parameters()]
        self.criterion = PredLoss(self.args)
        for data in self.datasets["valid"].keys():
            self.early_stopping_dict[data] = utils.EarlyStopping(
                patience=self.args.patience,
                compare=self.metric.compare,
                metric=self.metric.update_target,
            )

        if args.resume:
            save_dir = (
                os.path.join(
                    self.args.resume_dir,
                    self.args.exp_name,
                )
                + "/"
            )
            for c, data_name in enumerate(args.src_data):
                self.local_models[c].load_state_dict(
                    torch.load(
                        save_dir + self.args.save_prefix + f"last_{data_name}.pt",
                        map_location="cpu",
                    )
                )
            self.server_model.load_state_dict(
                torch.load(
                    save_dir + self.args.save_prefix + f"last_{data_name}.pt",
                    map_location="cpu",
                )
            )
            self.mertric = torch.load(
                save_dir + self.args.save_prefix + f"last_metric.pt", map_location="cpu"
            )
            self.early_stopping_dict = torch.load(
                save_dir + self.args.save_prefix + f"last_early.pt", map_location="cpu"
            )
            rmtree(save_dir)

    def load_dataset(self, split, dataname) -> None:
        return HierarchicalEHRDataset(
            data=dataname,
            input_path=self.args.input_path,
            split=split,
            ratio=self.args.ratio,
            pred_target=self.args.pred_target,
            seed=self.args.seed,
            debug=self.args.debug,
            max_word_len=self.args.max_word_len,
            max_seq_len=self.args.max_seq_len,
        )

    def dataloader_set(self, dataset, world_size, batch_size):
        if 1 < world_size:
            sampler = DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset,
                collate_fn=dataset.collator,
                batch_size=batch_size,
                num_workers=8,
                sampler=sampler,
                pin_memory=True,
            )
        else:
            sampler = None
            data_loader = DataLoader(
                dataset,
                collate_fn=dataset.collator,
                batch_size=batch_size,
                num_workers=8,
                shuffle=True,
                pin_memory=True,
            )
        return data_loader

    def setup_dist(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.args.port
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    @property
    def data_parallel_world_size(self):
        if self.args.world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.args.world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        return self.data_parallel_rank == 0

    def train(self):
        if 1 < self.args.world_size:
            mp.spawn(
                self.distributed_train,
                args=(self.args.world_size,),
                nprocs=self.args.world_size,
                join=True,
            )
        else:
            self.distributed_train(self.args.device_num, self.args.world_size)

    def send_model_to_device(
        self,
        model,
    ):
        if 1 < self.world_size:
            device = torch.device(
                f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
            )
            model = model.to(device)
            model = DistributedDataParallel(
                model, device_ids=[self.rank], find_unused_parameters=False
            )
        else:
            model = nn.DataParallel(model, device_ids=self.args.device_ids).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return model

    def distributed_train(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        if 1 < world_size:
            self.setup_dist(rank, world_size)
            torch.cuda.set_device(rank)

        # Wandb init
        if self.is_data_parallel_master and not self.args.debug:
            wandb.init(
                entity=self.args.wandb_entity_name,
                project=self.args.wandb_project_name,
                name=self.args.exp_name,
                config=self.args,
                reinit=True,
                id=self.args.exp_name[:100],
            )

        self.train_loaders = [
            self.dataloader_set(dataset, self.args.world_size, self.args.batch_size)
            for dataset in self.datasets["train"].values()
        ]
        self.valid_loaders = [
            self.dataloader_set(dataset, self.args.world_size, self.args.batch_size)
            for dataset in self.datasets["valid"].values()
        ]
        self.test_loaders = [
            self.dataloader_set(dataset, self.args.world_size, self.args.batch_size)
            for dataset in self.datasets["test"].values()
        ]

        logger.debug("Before Training", torch.cuda.memory_reserved())

        for comms in range(self.args.communications):
            logger.info("[Comms] {}".format(comms))
            # Save prev.model for backup
            self.old_server = deepcopy(self.server_model)
            self.old_local = deepcopy(self.local_models)
            self.old_metric = deepcopy(self.metric)
            self.old_early = deepcopy(self.early_stopping_dict)

            for c, data_name in enumerate(self.train_data):
                model, train_loader = (
                    self.send_model_to_device(self.local_models[c]),
                    self.train_loaders[c],
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
                model.train()
                torch.cuda.empty_cache()
                logger.debug(
                    "Before start local training", torch.cuda.memory_reserved()
                )

                for epoch in range(self.args.local_epochs):
                    total_steps = epoch + comms * self.args.local_epochs

                    if (
                        self.args.algorithm == "fedprox"
                        or self.args.algorithm == "fedpxn"
                    ) and comms > 0:
                        train_results = train_fedprox(
                            self.args,
                            self.server_model,
                            model,
                            optimizer,
                            self.criterion,
                            total_steps,
                            train_loader,
                            self.metric,
                            data_name,
                            self.param_name,
                            self.is_data_parallel_master,
                            self.world_size,
                        )
                    else:  # FedAvg, FedBN
                        train_results = train_naive(
                            self.args,
                            model,
                            optimizer,
                            self.criterion,
                            total_steps,
                            train_loader,
                            self.metric,
                            data_name,
                            self.is_data_parallel_master,
                            self.world_size,
                        )
                    logger.debug("In local training", torch.cuda.memory_reserved())

                self.local_models[c] = model.module.to("cpu")
                del model
                del optimizer
                torch.cuda.empty_cache()
            # Communication
            logger.debug("Immediate before comm", torch.cuda.memory_reserved())
            if self.is_data_parallel_master:
                self.server_model, self.local_models = communication(
                    self.args, self.server_model, self.local_models, self.client_weights
                )

            self.server_model = [self.server_model]
            if self.world_size > 1:
                torch.distributed.broadcast_object_list(self.local_models, 0)
                torch.distributed.broadcast_object_list(self.server_model, 0)
            self.server_model = self.server_model[0]
            logger.debug("After sync models", torch.cuda.memory_reserved())
            total_steps = (comms + 1) * self.args.local_epochs

            # Validation

            stop_list = []
            for c, data_name in enumerate(self.train_data):
                model, valid_loader = (
                    self.send_model_to_device(self.local_models[c]),
                    self.valid_loaders[c],
                )
                model.eval()
                metric_dict = inference(
                    self.args,
                    model,
                    valid_loader,
                    "valid",
                    data_name,
                    total_steps,
                    self.criterion,
                    self.metric,
                    self.is_data_parallel_master,
                    self.world_size,
                )

                if self.early_stopping_dict[data_name](
                    metric_dict[self.metric.update_target]
                ):
                    if self.is_data_parallel_master:
                        best_model_path = os.path.join(
                            self.args.save_dir,
                            self.args.exp_name,
                            self.args.save_prefix + f"_{data_name}_best.pt",
                        )
                        torch.save(self.local_models[c].state_dict(), best_model_path)
                del model
                torch.cuda.empty_cache()
                if self.early_stopping_dict[data_name].early_stop:
                    logger.info(f"data_name : {data_name}, Early stopping!")
                    stop_list.append(c)
            if self.is_data_parallel_master and self.args.debug == False:
                wandb.log({"comms": comms}, commit=True)

            if len(stop_list) == len(self.train_data):
                logger.info(f"all valid finished at {comms}")
                break
        if self.data_parallel_world_size > 1:
            dist.barrier()

        for c, data_name in enumerate(self.train_data):
            self.local_models[c].load_state_dict(
                torch.load(
                    os.path.join(
                        self.args.save_dir,
                        self.args.exp_name,
                        self.args.save_prefix + f"_{data_name}_best.pt",
                    ),
                    map_location="cpu",
                )
            )

        for c, data_name in enumerate(self.train_data):
            model, test_loader = (
                self.send_model_to_device(self.local_models[c]),
                self.test_loaders[c],
            )
            model.eval()
            metric_dict = inference(
                self.args,
                model,
                test_loader,
                "test",
                data_name,
                total_steps,
                self.criterion,
                self.metric,
                self.is_data_parallel_master,
                self.world_size,
            )
            self.local_models[c] = model.module.to("cpu")
        if self.is_data_parallel_master and self.args.debug == False:
            wandb.log({"comms": -1})
            wandb.finish(0)

        if self.data_parallel_world_size > 1:
            dist.destroy_process_group()

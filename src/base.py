import os
import wandb
from itertools import chain
from typing import Any, Dict, List
import logging

import numpy as np
import torch
import torch.nn as nn
from .model import UniHPF
from .metric import PredMetric

from .utils import trainer_utils as utils
from .utils import distributed_utils
from torch.utils.data import DataLoader
from .loss import PredLoss
from .dataset import HierarchicalEHRDataset
import tqdm

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import random
from collections import OrderedDict

logger = logging.getLogger(__name__)


def base(args):
    if args.train_type in ["pooled", "federated"] and len(args.src_data) == 1:
        raise AssertionError("pooled must select at least two datasets")

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
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True

    trainer = BaseTrainer(args, args.seed)
    trainer.train()
    logger.info("done training")


class BaseTrainer:
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed
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

    def dataloader_set(self, dataset, world_size, batch_size, collator):
        if 1 < world_size:
            self.sampler = DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset,
                collate_fn=collator,
                batch_size=batch_size,
                num_workers=8,
                sampler=self.sampler,
                pin_memory=True,
            )
        else:
            self.sampler = None
            data_loader = DataLoader(
                dataset,
                collate_fn=collator,
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

    def distributed_train(self, rank, world_size):
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
            )

        model = UniHPF(self.args)

        if 1 < world_size:
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )
            self.model = model.to(device)
            self.model = DistributedDataParallel(
                self.model, device_ids=[rank], find_unused_parameters=False
            )
        else:
            self.model = nn.DataParallel(model, device_ids=self.args.device_ids).to(
                "cuda"
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        logger.info(f"device_ids = {self.args.device_ids}")

        # Use ConcatDataset
        data_loader = self.dataloader_set(
            torch.utils.data.ConcatDataset(self.datasets["train"].values()),
            world_size,
            self.args.batch_size,
            collator=list(self.datasets["train"].values())[0].collator,
        )

        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)

        for data in self.datasets["valid"].keys():
            self.early_stopping_dict[data] = utils.EarlyStopping(
                patience=self.args.patience,
                compare=self.metric.compare,
                metric=self.metric.update_target,
            )

        break_token = False
        if not self.args.ratio == "0":
            for n_epoch in range(self.args.communications):

                logger.info("[Epoch] {}".format(n_epoch))
                self.model.train()

                for iter, sample in tqdm.tqdm(enumerate(data_loader)):
                    self.optimizer.zero_grad(set_to_none=True)

                    output = self.model(**sample["net_input"])
                    target = self.model.module.get_targets(sample)

                    loss = self.criterion(output, target)
                    loss.backward()

                    self.optimizer.step()

                    preds = torch.sigmoid(output["pred_output"]).view(-1).detach()
                    truths = target.view(-1)

                    logging_outputs = {
                        "loss": float(loss.detach().cpu()),
                        "preds": preds,
                        "truths": truths,
                    }
                    if self.data_parallel_world_size > 1:
                        _logging_outputs = self._all_gather_list_sync([logging_outputs])

                        for key in logging_outputs.keys():
                            if key == "loss":
                                logging_outputs[key] = float(
                                    sum(log[key] for log in _logging_outputs)
                                )
                            elif key in ["preds", "truths"]:
                                logging_outputs[key] = np.concatenate(
                                    [log[key].numpy() for log in _logging_outputs]
                                )
                            else:
                                raise NotImplementedError("What else?")
                        del _logging_outputs
                    else:
                        logging_outputs["preds"] = (
                            logging_outputs["preds"].cpu().numpy()
                        )
                        logging_outputs["truths"] = (
                            logging_outputs["truths"].cpu().numpy()
                        )

                    self.metric(**logging_outputs)  # iter_uddate

                with torch.no_grad():
                    train_metric_dict = self.metric.get_epoch_dict(len(data_loader))

                log_dict = utils.log_from_dict(
                    train_metric_dict, "train", self.train_data, n_epoch
                )

                if self.is_data_parallel_master and self.args.debug == False:
                    wandb.log(log_dict)

                break_token = self.evaluation(n_epoch)

                if break_token:
                    break
        if self.data_parallel_world_size > 1:
            dist.barrier()
        self.test(n_epoch)
        print(f"test finished at epoch {n_epoch}")

        if self.is_data_parallel_master and self.args.debug == False:
            wandb.finish(0)

        if self.data_parallel_world_size > 1:
            dist.destroy_process_group()

    def inference(self, data_loader, data_type, data_name, n_epoch):
        self.model.eval()
        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(data_loader)):
                self.optimizer.zero_grad(set_to_none=True)

                output = self.model(**sample["net_input"])
                target = self.model.module.get_targets(sample)

                loss = self.criterion(output, target)

                preds = torch.sigmoid(output["pred_output"]).view(-1).detach()
                truths = target.view(-1)

                logging_outputs = {
                    "loss": float(loss.detach().cpu()),
                    "preds": preds,
                    "truths": truths,
                }
                if self.data_parallel_world_size > 1:
                    _logging_outputs = self._all_gather_list_sync([logging_outputs])

                    for key in logging_outputs.keys():
                        if key == "loss":
                            logging_outputs[key] = float(
                                sum(log[key] for log in _logging_outputs)
                            )
                        elif key in ["preds", "truths"]:
                            logging_outputs[key] = np.concatenate(
                                [log[key].numpy() for log in _logging_outputs]
                            )
                        else:
                            raise NotImplementedError("What else?")
                    del _logging_outputs
                else:
                    logging_outputs["preds"] = logging_outputs["preds"].cpu().numpy()
                    logging_outputs["truths"] = logging_outputs["truths"].cpu().numpy()

                self.metric(**logging_outputs)  # iter_uddate

            metric_dict = self.metric.get_epoch_dict(len(data_loader))

        log_dict = utils.log_from_dict(metric_dict, data_type, data_name, n_epoch)

        if self.is_data_parallel_master and self.args.debug == False:
            wandb.log(log_dict)

        return metric_dict

    def evaluation(self, n_epoch):
        self.model.eval()
        break_token = False
        stop_list = []

        for c, data_name in enumerate(self.datasets["valid"].keys()):
            dataset = self.datasets["valid"][data_name]
            data_loader = self.dataloader_set(
                dataset,
                self.data_parallel_world_size,
                self.args.batch_size,
                dataset.collator,
            )
            metric_dict = self.inference(data_loader, "valid", data_name, n_epoch)

            if self.early_stopping_dict[data_name](
                metric_dict[self.metric.update_target]
            ):
                if self.is_data_parallel_master:
                    best_model_path = os.path.join(
                        self.args.save_dir,
                        self.args.exp_name,
                        self.args.save_prefix + f"_{c}_{data_name}_best.pt",
                    )
                    utils.model_save(
                        self.model, best_model_path, n_epoch, self.optimizer
                    )

            if self.early_stopping_dict[data_name].early_stop:
                logger.info(f"data_name : {data_name}, Early stopping!")
                stop_list.append(data_name)

        if len(self.datasets["valid"]) == len(stop_list):
            break_token = True
            logger.info(f"all valid finished at {n_epoch}")

        return break_token

    def test(self, n_epoch, load_checkpoint=None):
        print("test start .. ")
        if load_checkpoint is not None:
            for c, data_name in self.datasets["test"].keys():
                dataset = self.datasets["test"][data_name]
                load_path = load_checkpoint
                state_dict = torch.load(load_path, map_location="cpu")[
                    "model_state_dict"
                ]
                self.model.load_state_dict(state_dict, strict=True)

                data_loader = self.dataloader_set(
                    dataset,
                    self.data_parallel_world_size,
                    self.args.batch_size,
                    dataset.collator,
                )
                metric_dict = self.inference(data_loader, "test", data_name, n_epoch)
        else:
            for c, data_name in enumerate(self.datasets["test"].keys()):
                dataset = self.datasets["test"][data_name]
                best_model_path = os.path.join(
                    self.args.save_dir,
                    self.args.exp_name,
                    self.args.save_prefix + f"_{c}_{data_name}_best.pt",
                )
                state_dict = torch.load(best_model_path, map_location="cpu")[
                    "model_state_dict"
                ]
                self.model.load_state_dict(state_dict, strict=True)

                data_loader = self.dataloader_set(
                    dataset,
                    self.data_parallel_world_size,
                    self.args.batch_size,
                    dataset.collator,
                )
                metric_dict = self.inference(data_loader, "test", data_name, n_epoch)

        return metric_dict

    def _all_gather_list_sync(
        self, logging_outputs: List[Dict[str, Any]], ignore=False
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suifeature when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs],
                    max_size=getattr(self, "all_gather_list_size", 1048576),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs = results[0]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        return logging_outputs

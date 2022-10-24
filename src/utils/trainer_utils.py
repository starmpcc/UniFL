import logging
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=True, delta=0, compare="increase", metric="auprc"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metric_min = 0
        self.delta = delta
        self.compare_score = self.increase if compare == "increase" else self.decrease
        self.metric = metric

    def __call__(self, target_metric):
        update_token = False
        score = target_metric

        if self.best_score is None:
            self.best_score = score

        if self.compare_score(score):
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                logger.info(
                    f"Validation {self.metric} {self.compare_score.__name__}d {self.target_metric_min:.6f} --> {target_metric:.6f})"
                )
            self.target_metric_min = target_metric
            self.counter = 0
            update_token = True

        return update_token

    def increase(self, score):
        if score < self.best_score + self.delta:
            return True
        else:
            return False

    def decrease(self, score):
        if score > self.best_score + self.delta:
            return True
        else:
            return False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_save(model, path, n_epoch, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict()
            if (  # server model case
                isinstance(model, DataParallel)
                or isinstance(model, DistributedDataParallel)
            )
            else model.state_dict(),  # client model case
            "n_epoch": n_epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info(f"model save at : {path}")


def log_from_dict(metric_dict, data_type, data_name, n_epoch, is_master=False):
    if isinstance(data_name, list):
        data_name = "_".join(data_name)
    log_dict = {"epoch": n_epoch}
    for metric, values in metric_dict.items():
        log_dict[data_type + "/" + data_name + "_" + metric] = values
        if is_master:
            logger.info(
                data_type + "/" + data_name + "_" + metric + " : {:.3f}".format(values)
            )
    return log_dict

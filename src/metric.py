from sklearn.metrics import average_precision_score
import numpy as np


class PredMetric:
    def __init__(self, args, target="auprc"):
        self._update_target = target
        self.reset()

        self.pred_target = args.pred_target

    def reset(self):
        self.loss = 0
        self.truths = []
        self.preds = []

    def __call__(self, loss: float, preds: np.ndarray, truths: np.ndarray, **kwargs):
        self.truths += list(truths)
        self.preds += list(preds)
        self.loss += loss

    def get_epoch_dict(self, total_iter):
        self.epoch_dict = {
            "auprc": self.auprc,
            "loss": self.loss / total_iter,
        }
        self.reset()

        return self.epoch_dict

    @property
    def compare(self):
        return "decrease" if self.update_target == "loss" else "increase"

    @property
    def update_target(self):
        return self._update_target

    @property
    def auprc(self):
        return average_precision_score(self.truths, self.preds, average="micro")

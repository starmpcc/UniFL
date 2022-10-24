import torch.nn as nn


class PredLoss:
    def __init__(self, args):
        self.args = args
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.args = args

    def __call__(self, output, target):
        if self.args.pred_target in ["mort", "los3", "los7", "readm"]:
            return self.binary_class(output, target)
        elif self.args.pred_target in ["dx"]:
            return self.multi_label_multi_class(output, target)

    def binary_class(self, output, target):
        return self.BCE_loss(
            output["pred_output"], target.squeeze(dim=-1).to(output["pred_output"].device)
        )

    def multi_label_multi_class(self, output, target):
        return self.BCE_loss(
            output["pred_output"].view(-1),
            target.view(-1).to(output["pred_output"].device),
        )

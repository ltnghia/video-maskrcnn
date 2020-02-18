# Wing loss for Robust Facial Landmark Localisation with Convolutional Neural Networks, CVPR 2018

import torch
import numpy as np
import torch.nn as nn
from .utils import add_weights_and_normalize


class WingLoss(nn.Module):
    def __init__(self, width=10, curvature=2):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)

    def forward(self, prediction, target, weight=None, reduction='mean'):
        diff_abs = torch.abs(target - prediction)
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger] = loss[idx_bigger] - self.C

        return add_weights_and_normalize(loss,
                                         label=target,
                                         weight=weight,
                                         reduction=reduction)


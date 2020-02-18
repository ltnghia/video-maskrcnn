# Libra R-CNN: Towards Balanced Learning for Object Detection, CVPR 2019

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .utils import add_weights_and_normalize


class BalancedL1Loss(nn.Module):
    """Balanced L1 loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred, target, weight=None, reduction='mean'):
        loss_bbox = self.balanced_l1_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta)

        return add_weights_and_normalize(loss_bbox,
                                         label=target,
                                         weight=weight,
                                         reduction=reduction)

    def balanced_l1_loss(self,
                         pred,
                         target,
                         beta=1.0,
                         alpha=0.5,
                         gamma=1.5,
                         reduction='none'):
        assert beta > 0
        # print(pred.size(), target.size(), target.numel())
        assert pred.size() == target.size() and target.numel() > 0

        diff = torch.abs(pred - target)

        b = np.e ** (gamma / alpha) - 1
        loss = torch.where(
            diff < beta, alpha / b *
            (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
            gamma * diff + gamma / b - alpha * beta)

        return loss
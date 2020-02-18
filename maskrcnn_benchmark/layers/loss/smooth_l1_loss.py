# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from .utils import add_weights_and_normalize


class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta=1. / 9):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target, weight=None, reduction='mean'):
        return smooth_l1_loss(input, target, self.beta, weight=weight, reduction=reduction)


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, weight=None, reduction='mean'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return add_weights_and_normalize(loss,
                                     label=target,
                                     weight=weight,
                                     reduction=reduction)

import torch
from .utils import add_weights_and_normalize


def l2_loss(input, target, weight=None, reduction='mean'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond**2 / pos_inds.shape[0]
    else:
        loss = input * 0.0

    return add_weights_and_normalize(loss,
                                     label=target[pos_inds],
                                     weight=weight,
                                     reduction=reduction)


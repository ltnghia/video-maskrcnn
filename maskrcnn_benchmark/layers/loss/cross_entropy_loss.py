import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import _expand_binary_labels, add_weights_and_normalize


def cross_entropy(pred, label, weight=None, reduction='mean'):
    loss = F.cross_entropy(pred, label, reduction='none')
    return add_weights_and_normalize(loss,
                                     label=label,
                                     weight=weight,
                                     reduction=reduction)


def binary_cross_entropy(pred, label, weight=None, reduction='mean'):
    loss = F.binary_cross_entropy(pred, label, reduction='none')
    return add_weights_and_normalize(loss,
                                     label=label,
                                     weight=weight,
                                     reduction=reduction)


def weighted_binary_cross_entropy(pred, label, weight=None, reduction='mean'):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight.float())
    return add_weights_and_normalize(loss,
                                     label=label,
                                     weight=weight,
                                     reduction=reduction)


def mask_cross_entropy(pred, target, label, weight=None, reduction='mean'):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    loss = F.binary_cross_entropy_with_logits(pred_slice, target, reduction='mean')
    return add_weights_and_normalize(loss,
                                     label=label,
                                     weight=weight,
                                     reduction=reduction)


# Improving Object Localization with Fitness NMS and Bounded IoU loss

import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import bbox_overlaps, add_weights_and_normalize


class IOULoss(nn.Module):
    def __init__(self, style='naive', beta=0.2, eps=1e-3):
        self.style = style
        self.beta = beta
        self.eps = eps
        if style not in ['bounded', 'naive']:
            raise ValueError('Only support bounded iou loss and naive iou loss.')

    def forward(self, pred, target, weight=None, reduction='mean'):
        if self.style == 'bounded':
            return bounded_iou_loss(pred, target, beta=self.beta, eps=self.eps, reduction='none')
        else:
            return iou_loss(pred, target, weight=weight, reduction='none')


def iou_loss(pred_bboxes, target_bboxes, weight=None, reduction='mean'):
    ious = bbox_overlaps(pred_bboxes, target_bboxes, is_aligned=True)
    loss = -ious.log()
    return add_weights_and_normalize(loss,
                                     label=target_bboxes,
                                     weight=weight,
                                     reduction=reduction)


def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3, weight=None, reduction='mean'):
    """Improving Object Localization with Fitness NMS and Bounded IoU loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
        reduction (str): Reduction type.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)

    return add_weights_and_normalize(loss,
                                     label=target,
                                     weight=weight,
                                     reduction=reduction)



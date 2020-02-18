# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable

from .utils import concat_box_prediction_layers
from maskrcnn_benchmark.modeling.sampler.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler

from maskrcnn_benchmark.modeling.matcher.matcher import Matcher
from maskrcnn_benchmark.structures.ops import boxlist_iou
from maskrcnn_benchmark.structures.ops import cat_boxlist

from maskrcnn_benchmark.layers import SmoothL1Loss
from maskrcnn_benchmark.layers import WingLoss
from maskrcnn_benchmark.layers import BalancedL1Loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.layers import AdjustSmoothL1Loss


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, generate_labels_func,
                 wing_loss=None, adjust_smooth_l1_loss=None, balance_l1_loss=None,
                 focal_loss=None, combination_weight=0, use_negative_samples=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']
        self.wing_loss = wing_loss
        self.focal_loss = focal_loss
        self.combination_weight = combination_weight
        self.adjust_smooth_l1_loss = adjust_smooth_l1_loss
        self.balance_l1_loss = balance_l1_loss
        self.smooth_l1_loss = SmoothL1Loss(beta=1.0 / 9)
        self.use_negative_samples = use_negative_samples

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single GT in the image,
        # and matched_idxs can be -2, which goes out of bounds

        # matched_targets = target[matched_idxs.clamp(min=0)]

        if self.use_negative_samples and len(target) == 0:
            matched_targets = target
        else:
            matched_targets = target[matched_idxs.clamp(min=0)]

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(anchors_per_image, targets_per_image, self.copied_fields)

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)

            # compute regression targets
            if self.use_negative_samples and len(matched_targets) == 0:
                zeros = torch.zeros_like(labels_per_image)
                regression_targets_per_image = torch.stack((zeros, zeros, zeros, zeros), dim=1)
            else:
                regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[list[BoxList]])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        if targets is None:
            return None, None

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        labels, regression_targets = self.prepare_targets(anchors, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # print(sampled_pos_inds)

        if self.balance_l1_loss:
            box_loss = self.balance_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                reduction='sum',
            )
        elif self.adjust_smooth_l1_loss:
            box_loss = self.adjust_smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                reduction='sum',
            ) / 4
        elif self.wing_loss:
            box_loss = self.wing_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                reduction='sum',
            )
        else:
            box_loss = self.smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                reduction='sum',
            )

        if len(sampled_inds) == 0:
            box_loss = box_loss * 0
        else:
            box_loss = box_loss / sampled_inds.numel()

        if self.focal_loss:
            if self.combination_weight == 0:
                if len(labels.shape) == 1:
                    objectness = objectness.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    labels = labels.to(dtype=torch.int32)

                objectness_loss = self.focal_loss(objectness, labels, reduction='sum')
                objectness_loss = objectness_loss / (sampled_inds.numel() + labels.shape[0])
            else:
                bce_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], reduction='mean')
                if len(labels.shape) == 1:
                    objectness = objectness.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    labels = labels.to(dtype=torch.int32)

                fc_loss = self.focal_loss(objectness, labels, reduction='sum')
                fc_loss = fc_loss / (sampled_inds.numel() + labels.shape[0])
                objectness_loss = (fc_loss + bce_loss) * self.combination_weight
        else:
            objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], reduction='mean')

        return objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    if cfg.MODEL.RPN.USE_WING_LOSS:
        wing_loss = WingLoss(width=cfg.MODEL.RPN.WING_LOSS.WIDTH,
                             curvature=cfg.MODEL.RPN.WING_LOSS.SIGMA,)
    else:
        wing_loss = None

    if cfg.MODEL.RPN.USE_FOCAL_LOSS:
        focal_loss = SigmoidFocalLoss(
            cfg.MODEL.RPN.FOCAL_LOSS.GAMMA,
            cfg.MODEL.RPN.FOCAL_LOSS.ALPHA,
        )
    else:
        focal_loss = None

    if cfg.MODEL.RPN.USE_SELF_ADJUST_SMOOTH_L1_LOSS:
        adjust_smooth_l1_loss = AdjustSmoothL1Loss(
            4,
            beta=cfg.MODEL.RPN.SELF_ADJUST_SMOOTH_L1_LOSS.BBOX_REG_BETA
        )
    else:
        adjust_smooth_l1_loss = None

    if cfg.MODEL.RPN.USE_COMBINATION_LOSS:
        combination_weight = cfg.MODEL.RPN.COMBINATION_LOSS.WEIGHT
    else:
        combination_weight = 0

    if cfg.MODEL.RPN.USE_BALANCE_L1_LOSS:
        balance_l1_loss = BalancedL1Loss(alpha=cfg.MODEL.RPN.BALANCE_L1_LOSS.ALPHA,
                                         beta=cfg.MODEL.RPN.BALANCE_L1_LOSS.BETA,
                                         gamma=cfg.MODEL.RPN.BALANCE_L1_LOSS.GAMMA)
    else:
        balance_l1_loss = None

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels,
        wing_loss=wing_loss,
        adjust_smooth_l1_loss=adjust_smooth_l1_loss,
        balance_l1_loss=balance_l1_loss,
        focal_loss=focal_loss,
        combination_weight=combination_weight
    )
    return loss_evaluator

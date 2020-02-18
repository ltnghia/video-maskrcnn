# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import os

from maskrcnn_benchmark.layers import SmoothL1Loss
from maskrcnn_benchmark.layers import WingLoss
from maskrcnn_benchmark.layers import AdjustSmoothL1Loss
from maskrcnn_benchmark.layers import BalancedL1Loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss, cross_entropy
from maskrcnn_benchmark.layers import ClassBalanceLoss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher.matcher import Matcher
from maskrcnn_benchmark.structures.ops import boxlist_iou
from maskrcnn_benchmark.modeling.sampler.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False,
            focal_loss=None,
            class_balance_weight=None,
            wing_loss=None,
            adjust_smooth_l1_loss=None,
            balance_l1_loss=None,
            use_negative_samples=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.focal_loss = focal_loss
        self.class_balance_loss = class_balance_weight
        self.wing_loss = wing_loss
        self.adjust_smooth_l1_loss = adjust_smooth_l1_loss
        self.balance_l1_loss = balance_l1_loss
        self.smooth_l1_loss = SmoothL1Loss()
        self.use_negative_samples = use_negative_samples

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        # target = target.copy_with_fields("labels")
        # matched_targets = target[matched_idxs.clamp(min=0)]

        if self.use_negative_samples and len(target) == 0:
            matched_targets = target
            matched_targets.add_field("labels", matched_idxs.clamp(min=1, max=1))
        else:
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields("labels")
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single GT in the image,
            # and matched_idxs can be -2, which goes out of bounds
            matched_targets = target[matched_idxs.clamp(min=0)]

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, proposals_per_image.bbox)

            # compute regression targets
            if self.use_negative_samples and len(matched_targets) == 0:
                zeros = torch.zeros_like(labels_per_image, dtype=torch.float)
                regression_targets_per_image = torch.stack((zeros, zeros, zeros, zeros), dim=1)
            else:
                regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, proposals_per_image.bbox)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            return None, None
            # raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        if self.class_balance_loss:
            class_balance_weight = self.class_balance_loss.get_weights(labels)
        else:
            class_balance_weight = None

        if self.focal_loss:
            classification_loss = self.focal_loss(class_logits,
                                                  labels.int(),
                                                  weight=class_balance_weight,
                                                  reduction='mean')
        else:
            classification_loss = cross_entropy(class_logits,
                                                labels,
                                                weight=class_balance_weight,
                                                reduction='mean')

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            labels_pos = labels[sampled_pos_inds_subset]
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        if self.balance_l1_loss:
            box_loss = self.balance_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                reduction='sum',
            )
        elif self.adjust_smooth_l1_loss:
            box_loss = self.adjust_smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                reduction='sum',
            ) / 4
        elif self.wing_loss:
            box_loss = self.wing_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                reduction='sum',
            )
        else:
            box_loss = self.smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                reduction='sum',
            )

        if len(labels) == 0:
            box_loss = box_loss * 0
        else:
            box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    if cfg.MODEL.ROI_BOX_HEAD.USE_FOCAL_LOSS:
        focal_loss = SigmoidFocalLoss(
            cfg.MODEL.ROI_BOX_HEAD.FOCAL_LOSS.GAMMA,
            cfg.MODEL.ROI_BOX_HEAD.FOCAL_LOSS.ALPHA
        )
        # focal_loss = SoftmaxFocalLoss(
        #     class_num = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES-1,
        #     gamma=cfg.MODEL.RPN.FOCAL_LOSS.GAMMA,
        #     alpha=cfg.MODEL.RPN.FOCAL_LOSS.ALPHA,
        # )
    else:
        focal_loss = None

    if cfg.MODEL.ROI_BOX_HEAD.USE_CLASS_BALANCE_LOSS and \
        os.path.isfile(cfg.MODEL.ROI_BOX_HEAD.CLASS_BALANCE_LOSS.WEIGHT_FILE):
        num_class_list = ClassBalanceLoss.load_class_samples(filename=cfg.MODEL.ROI_BOX_HEAD.CLASS_BALANCE_LOSS.WEIGHT_FILE,
                                                             category_type='category')
        class_balance_weight = ClassBalanceLoss(device=torch.device(cfg.MODEL.DEVICE),
                                                num_class_list=num_class_list,
                                                alpha=cfg.MODEL.ROI_BOX_HEAD.CLASS_BALANCE_LOSS.ALPHA,
                                                beta=cfg.MODEL.ROI_BOX_HEAD.CLASS_BALANCE_LOSS.BETA)
    else:
        class_balance_weight = None

    if cfg.MODEL.ROI_BOX_HEAD.USE_WING_LOSS:
        wing_loss = WingLoss(width=cfg.MODEL.ROI_BOX_HEAD.WING_LOSS.WIDTH,
                             curvature=cfg.MODEL.ROI_BOX_HEAD.WING_LOSS.SIGMA,)
    else:
        wing_loss = None

    if cfg.MODEL.ROI_BOX_HEAD.USE_SELF_ADJUST_SMOOTH_L1_LOSS:
        adjust_smooth_l1_loss = AdjustSmoothL1Loss(
            4,
            beta=cfg.MODEL.ROI_BOX_HEAD.SELF_ADJUST_SMOOTH_L1_LOSS.BBOX_REG_BETA
        )
    else:
        adjust_smooth_l1_loss = None

    if cfg.MODEL.ROI_BOX_HEAD.USE_BALANCE_L1_LOSS:
        balance_l1_loss = BalancedL1Loss(alpha=cfg.MODEL.ROI_BOX_HEAD.BALANCE_L1_LOSS.ALPHA,
                                         beta=cfg.MODEL.ROI_BOX_HEAD.BALANCE_L1_LOSS.BETA,
                                         gamma=cfg.MODEL.ROI_BOX_HEAD.BALANCE_L1_LOSS.GAMMA)
    else:
        balance_l1_loss = None

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg,
        focal_loss=focal_loss,
        class_balance_weight=class_balance_weight,
        wing_loss=wing_loss,
        adjust_smooth_l1_loss=adjust_smooth_l1_loss,
        balance_l1_loss=balance_l1_loss,
    )

    return loss_evaluator

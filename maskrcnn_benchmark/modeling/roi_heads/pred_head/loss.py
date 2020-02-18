import torch
import os

from maskrcnn_benchmark.modeling.matcher.matcher import Matcher
from maskrcnn_benchmark.layers import SigmoidFocalLoss, cross_entropy, ClassBalanceLoss
from maskrcnn_benchmark.structures.ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class PredRCNNLossComputation(object):
    def __init__(self, proposal_matcher, focal_loss=None, class_balance_loss=None, use_negative_samples=False):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.proposal_matcher = proposal_matcher
        self.focal_loss = focal_loss
        self.class_balance_loss = class_balance_loss
        self.use_negative_samples = use_negative_samples

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        # target = target.copy_with_fields(["labels", "pred_labels"])
        # matched_targets = target[matched_idxs.clamp(min=0)]

        if self.use_negative_samples and len(target) == 0:
            matched_targets = target
            matched_targets.add_field("labels", matched_idxs.clamp(min=1, max=1))
            matched_targets.add_field("pred_labels", matched_idxs.clamp(min=1, max=1))
        else:
            # pred RCNN needs "labels" and "pred_labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "pred_labels"])
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single GT in the image,
            # and matched_idxs can be -2, which goes out of bounds
            matched_targets = target[matched_idxs.clamp(min=0)]

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        preds = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            # this can probably be removed, but is left here for clarity and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
            # pred_labels = matched_targets.get_field("pred_labels")
            # preds_per_image = pred_labels[positive_inds]

            if self.use_negative_samples and len(matched_targets) == 0:
                zeros = torch.zeros_like(labels_per_image, dtype=torch.float)
                preds_per_image = torch.stack((zeros, zeros, zeros, zeros), dim=1)
            else:
                # pred scores are only computed on positive samples
                positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
                pred_labels = matched_targets.get_field("pred_labels")
                preds_per_image = pred_labels[positive_inds]

            labels.append(labels_per_image)
            preds.append(preds_per_image)

        return labels, preds

    def __call__(self, proposals, class_logits, targets):
        """
        Computes the loss for pred R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (Tensor)

        Returns:
            classification_loss (Tensor)
        """

        if targets is None:
            return None

        labels, pred_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        pred_targets = cat(pred_targets, dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't accept empty tensors, so handle it separately
        if pred_targets.numel() == 0:
            return class_logits.sum() * 0

        # get indices that correspond to the regression targets for the corresponding ground truth labels,
        # to be used with advanced indexing labels = [labels > 0]
        labels = pred_targets
        valid = torch.nonzero(labels > 0).squeeze(1)
        class_logits = class_logits[valid]
        labels = labels[valid]

        if self.class_balance_loss:
            class_balance_weight = self.class_balance_loss.get_weights(labels)
        else:
            class_balance_weight = None

        if self.focal_loss is not None:
            loss_pred = self.focal_loss(class_logits,
                                        labels.int(),
                                        weight=class_balance_weight,
                                        reduction='mean')
            # loss_pred = loss_pred / (valid.numel() + labels.numel())
        else:
            loss_pred = cross_entropy(class_logits,
                                      labels.long(),
                                      weight=class_balance_weight,
                                      reduction='mean')
        return loss_pred


def make_roi_pred_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    if cfg.MODEL.ROI_PRED_HEAD.USE_FOCAL_LOSS:
        focal_loss = SigmoidFocalLoss(
            cfg.MODEL.ROI_PRED_HEAD.FOCAL_LOSS.GAMMA,
            cfg.MODEL.ROI_PRED_HEAD.FOCAL_LOSS.ALPHA
        )
        # focal_loss = SoftmaxFocalLoss(
        #     class_num = 1,
        #     gamma=cfg.MODEL.RPN.FOCAL_LOSS.GAMMA,
        #     alpha=cfg.MODEL.RPN.FOCAL_LOSS.ALPHA,
        # )
    else:
        focal_loss = None

    if cfg.MODEL.ROI_PRED_HEAD.USE_CLASS_BALANCE_LOSS and \
        os.path.isfile(cfg.MODEL.ROI_PRED_HEAD.CLASS_BALANCE_LOSS.WEIGHT_FILE):
        num_class_list = ClassBalanceLoss.load_class_samples(filename=cfg.MODEL.ROI_PRED_HEAD.CLASS_BALANCE_LOSS.WEIGHT_FILE,
                                                             category_type='second_category')
        class_balance_loss = ClassBalanceLoss(device=torch.device(cfg.MODEL.DEVICE),
                                              num_class_list=num_class_list,
                                              alpha=cfg.MODEL.ROI_PRED_HEAD.CLASS_BALANCE_LOSS.ALPHA,
                                              beta=cfg.MODEL.ROI_PRED_HEAD.CLASS_BALANCE_LOSS.BETA)
    else:
        class_balance_loss = None

    loss_evaluator = PredRCNNLossComputation(
        proposal_matcher=matcher,
        focal_loss=focal_loss,
        class_balance_loss=class_balance_loss,
    )

    return loss_evaluator

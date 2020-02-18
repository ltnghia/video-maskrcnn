# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.boxes import BoxList
from maskrcnn_benchmark.modeling.utils import dfs_freeze
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from maskrcnn_benchmark.modeling.roi_heads.base_head import ROIBaseHead, keep_only_positive_boxes


class ROIMaskHead(ROIBaseHead):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__(cfg=cfg)
        self.use_extended_features = cfg.MODEL.ROI_MASK_HEAD.USE_EXTENDED_FEATURES
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        self.freeze = cfg.MODEL.ROI_MASK_HEAD.FREEZE

        if self.freeze:
            dfs_freeze(self, requires_grad=False)

    def forward(self, features, proposals, targets=None, global_features=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        all_proposals = proposals
        if self.training and targets is not None:
            # during training, only focus on positive boxes
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals, global_features=global_features)

        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)

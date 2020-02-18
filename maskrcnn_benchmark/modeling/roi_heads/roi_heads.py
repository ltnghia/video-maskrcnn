# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .pred_head.pred_head import build_roi_pred_head
from .maskiou_head.maskiou_head import build_roi_maskiou_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads, freeze=False, use_extended_features=False):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        self.freeze = freeze
        self.use_extended_features = use_extended_features
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.PREDICTION_ON and cfg.MODEL.ROI_PRED_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.prediction.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, extended_features=None, global_features=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        if extended_features is not None and self.box.use_extended_features:
            box_features = extended_features
        else:
            box_features = features
        x, detections, loss_box = self.box(box_features, proposals, targets, global_features=global_features)
        losses.update(loss_box)

        n_detection = 0
        for detection in detections:
            n_detection += len(detection)
        if n_detection <= 0:
            return x, detections, losses
        
        if self.cfg.MODEL.MASK_ON:
            if extended_features is not None and self.mask.use_extended_features:
                mask_features = extended_features
            else:
                mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if not self.cfg.MODEL.MASKIOU_ON:
                x, detections, loss_mask = self.mask(mask_features, detections, targets, global_features=global_features)
                losses.update(loss_mask)
            else:
                x, detections, loss_mask, roi_feature, selected_mask, labels, maskiou_targets = self.mask(mask_features,
                                                                                                          detections,
                                                                                                          targets)
                losses.update(loss_mask)
                loss_maskiou, detections = self.maskiou(roi_feature, detections, selected_mask, labels, maskiou_targets)
                losses.update(loss_maskiou)

        if self.cfg.MODEL.KEYPOINT_ON:
            if extended_features is not None and self.keypoint.use_extended_features:
                keypoint_features = extended_features
            else:
                keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if self.training and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets, global_features=global_features)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.PREDICTION_ON:
            if extended_features is not None and self.prediction.use_extended_features:
                prediction_features = extended_features
            else:
                prediction_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the prediction heads, then we can reuse the features already computed
            if self.training and self.cfg.MODEL.ROI_PRED_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                prediction_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_prediction = self.prediction(prediction_features, detections, targets, global_features=global_features)
            losses.update(loss_prediction)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    if cfg.MODEL.RPN_ONLY:
        return None

    if cfg.MODEL.RETINANET_ON:
        return None

    roi_heads = []
    if cfg.MODEL.BOX_ON:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
        if cfg.MODEL.MASKIOU_ON:
            roi_heads.append(("maskiou", build_roi_maskiou_head(cfg)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.PREDICTION_ON:
        roi_heads.append(("prediction", build_roi_pred_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        freeze = True
        use_extended_features = True
        for roi_head in roi_heads:
            freeze = (freeze and roi_head[1].freeze)
            use_extended_features = (use_extended_features and roi_head[1].use_extended_features)
        roi_heads = CombinedROIHeads(cfg, roi_heads, freeze=freeze, use_extended_features=use_extended_features)
        return roi_heads
    else:
        return None

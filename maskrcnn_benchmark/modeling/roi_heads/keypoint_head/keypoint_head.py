import torch

from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import make_roi_keypoint_predictor
from .inference import make_roi_keypoint_post_processor
from .loss import make_roi_keypoint_loss_evaluator
from maskrcnn_benchmark.modeling.roi_heads.base_head import ROIBaseHead
from maskrcnn_benchmark.modeling.utils import dfs_freeze


class ROIKeypointHead(ROIBaseHead):
    def __init__(self, cfg, in_channels):
        super(ROIKeypointHead, self).__init__(cfg=cfg)
        self.use_extended_features = cfg.MODEL.ROI_KEYPOINT_HEAD.USE_EXTENDED_FEATURES
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_keypoint_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

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
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.feature_extractor(features, proposals, global_features=global_features)
        kp_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(kp_logits, proposals)
            return x, result, {}

        loss_kp = self.loss_evaluator(proposals, kp_logits)

        return x, proposals, dict(loss_kp=loss_kp)


def build_roi_keypoint_head(cfg, in_channels):
    model = ROIKeypointHead(cfg, in_channels)

    if cfg.MODEL.ROI_KEYPOINT_HEAD.FREEZE:
        dfs_freeze(model, requires_grad=False)
    return model


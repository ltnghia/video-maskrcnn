from maskrcnn_benchmark.modeling import registry
from torch import nn
import math
import torch
from maskrcnn_benchmark.modeling.make_layers import make_fc


@registry.ROI_PRED_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = cfg.MODEL.ROI_PRED_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_classes > 0:
            self.pred_score = make_fc(num_inputs, num_classes, use_gn=False)
            # self.pred_score = nn.Linear(num_inputs, num_classes)
            #
            # nn.init.normal_(self.pred_score.weight, mean=0, std=0.01)
            # nn.init.constant_(self.pred_score.bias, 0)

            if cfg.MODEL.ROI_PRED_HEAD.USE_FOCAL_LOSS:
                # bias_init for sigmoid focal loss
                prior_prob = cfg.MODEL.ROI_PRED_HEAD.PRIOR_PROB
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(self.pred_score.bias, bias_value)
            elif cfg.MODEL.ROI_PRED_HEAD.USE_CLASS_BALANCE_LOSS:
                # bias_init for class balance loss
                bias_value = -math.log(num_classes - 1)
                nn.init.constant_(self.pred_score.bias, bias_value)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pred_logit = self.pred_score(x)
        return pred_logit


@registry.ROI_PRED_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_PRED_HEAD.NUM_CLASSES
        representation_size = in_channels

        if num_classes > 0:
            self.pred_score = make_fc(representation_size, num_classes, use_gn=False)
            # self.pred_score = nn.Linear(representation_size, num_classes)
            #
            # nn.init.normal_(self.pred_score.weight, std=0.01)
            # nn.init.constant_(self.pred_score.bias, 0)

            if cfg.MODEL.ROI_PRED_HEAD.USE_FOCAL_LOSS:
                # bias_init for sigmoid focal loss
                prior_prob = cfg.MODEL.ROI_PRED_HEAD.PRIOR_PROB
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(self.pred_score.bias, bias_value)
            elif cfg.MODEL.ROI_PRED_HEAD.USE_CLASS_BALANCE_LOSS:
                # bias_init for class balance loss
                bias_value = -math.log(num_classes - 1)
                nn.init.constant_(self.pred_score.bias, bias_value)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.pred_score(x)

        return scores


def make_roi_pred_predictor(cfg, in_channels):
    func = registry.ROI_PRED_PREDICTOR[cfg.MODEL.ROI_PRED_HEAD.PREDICTOR]
    return func(cfg, in_channels)

from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.pooler import make_pooler, make_contextual_pooler
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.attention_mechanism.regional_attention import RegionalAttention


@registry.ROI_KEYPOINT_FEATURE_EXTRACTORS.register("KeypointRCNNFeatureExtractor")
class KeypointRCNNFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(KeypointRCNNFeatureExtractor, self).__init__()

        use_gn = cfg.MODEL.ROI_KEYPOINT_HEAD.USE_GN

        use_contextual_pooler = False
        if use_contextual_pooler:
            pooler = make_contextual_pooler(cfg, 'ROI_KEYPOINT_HEAD')
        else:
            pooler = make_pooler(cfg, 'ROI_KEYPOINT_HEAD')
        self.pooler = pooler

        input_features = in_channels
        layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "kp_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=1, stride=1, use_gn=use_gn, use_relu=True, kaiming_init=True
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features
        if cfg.MODEL.ROI_KEYPOINT_HEAD.ATTENTION_ON:
            self.regional_attention = RegionalAttention(cfg, in_channels, self.pooler, resolution)
        else:
            self.regional_attention = None

    def forward(self, x, proposals, global_features=None):
        if self.regional_attention is not None:
            x = self.regional_attention(x, proposals, global_features)
        else:
            x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = getattr(self, layer_name)(x)
        return x


def make_roi_keypoint_feature_extractor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)

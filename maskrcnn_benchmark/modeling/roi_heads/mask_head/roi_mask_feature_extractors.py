# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.pooler import make_pooler, make_contextual_pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3, group_norm, make_blocks


registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        use_contextual_pooler = False
        if use_contextual_pooler:
            pooler = make_contextual_pooler(cfg, 'ROI_MASK_HEAD')
        else:
            pooler = make_pooler(cfg, 'ROI_MASK_HEAD')

        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION

        use_dcn = cfg.MODEL.ROI_MASK_HEAD.USE_DCN
        dcn_config = {
            "with_modulated_dcn": cfg.MODEL.ROI_MASK_HEAD.WITH_MODULATED_DCN,
            "deformable_groups": cfg.MODEL.ROI_MASK_HEAD.DEFORMABLE_GROUPS,
        }

        self.blocks, next_feature = make_blocks(layers, input_size,
                                                pattern='mask_fcn',
                                                parent=self,
                                                dilation=dilation,
                                                use_gn=use_gn,
                                                use_dcn=use_dcn,
                                                dcn_config=dcn_config,
                                                )

        self.out_channels = next_feature

        self.regional_attention = None

    def forward(self, x, proposals, global_features=None):
        if self.regional_attention is not None:
            x = self.regional_attention(x, proposals, global_features)
        else:
            x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = getattr(self, layer_name)(x)

        return x


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.pooler import make_pooler, make_contextual_pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm, make_fc, make_conv3x3, make_blocks


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        use_contextual_pooler = False
        if use_contextual_pooler:
            pooler = make_contextual_pooler(cfg, 'ROI_BOX_HEAD')
        else:
            pooler = make_pooler(cfg, 'ROI_BOX_HEAD')

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals, global_features=None):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        use_contextual_pooler = False
        if use_contextual_pooler:
            pooler = make_contextual_pooler(cfg, 'ROI_BOX_HEAD')
        else:
            pooler = make_pooler(cfg, 'ROI_BOX_HEAD')

        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals, global_features=None):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        use_contextual_pooler = False
        if use_contextual_pooler:
            pooler = make_contextual_pooler(cfg, 'ROI_BOX_HEAD')
        else:
            pooler = make_pooler(cfg, 'ROI_BOX_HEAD')

        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS

        use_dcn = cfg.MODEL.ROI_BOX_HEAD.USE_DCN
        dcn_config = {
            "with_modulated_dcn": cfg.MODEL.ROI_BOX_HEAD.WITH_MODULATED_DCN,
            "deformable_groups": cfg.MODEL.ROI_BOX_HEAD.DEFORMABLE_GROUPS,
        }

        self.xconvs, next_feature = make_blocks(layers, input_size,
                                                pattern='xconvs',
                                                parent=self,
                                                dilation=dilation,
                                                use_gn=use_gn,
                                                use_dcn=use_dcn,
                                                dcn_config=dcn_config,
                                                )

        input_size = next_feature * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

        if cfg.MODEL.ROI_BOX_HEAD.ATTENTION_ON:
            self.regional_attention = RegionalAttention(cfg, in_channels, self.pooler, resolution)
        else:
            self.regional_attention = None

    def forward(self, x, proposals, global_features=None):
        if self.regional_attention is not None:
            x = self.regional_attention(x, proposals, global_features)
        else:
            x = self.pooler(x, proposals)

        for layer_name in self.xconvs:
            x = getattr(self, layer_name)(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)

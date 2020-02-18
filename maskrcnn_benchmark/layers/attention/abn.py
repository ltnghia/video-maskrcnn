# attention Branch Network: Learning of attention Mechanism for Visual Explanation, CVPR 2019

import torch
import torch.nn as nn
from maskrcnn_benchmark.modeling.make_layers import make_conv1x1, make_blocks


class AttentionBranchNetwork(nn.Module):
    def __init__(self, cfg_attention, in_channels, num_classes):
        super(AttentionBranchNetwork, self).__init__()

        use_gn = cfg_attention.USE_GN
        dilation = cfg_attention.DILATION
        layers = cfg_attention.CONV_LAYERS
        use_dcn = cfg_attention.USE_DCN
        attention_type = cfg_attention.ATTENTION_TYPE
        dcn_config = {
            "with_modulated_dcn": cfg_attention.WITH_MODULATED_DCN,
            "deformable_groups": cfg_attention.DEFORMABLE_GROUPS,
        }

        self.att_block, next_feature = make_blocks(layers, in_channels,
                                                   pattern='attention_block',
                                                   parent=self,
                                                   dilation=dilation,
                                                   use_gn=use_gn,
                                                   use_dcn=use_dcn,
                                                   dcn_config=dcn_config, )

        self.att_conv = make_conv1x1(next_feature, num_classes, use_relu=False, kaiming_init=True, use_bias=True,
                                     use_gn=False, adaptive_group_norm=True)
        self.att_generator = make_conv1x1(num_classes, 1, use_relu=False, kaiming_init=True, use_bias=True,
                                          use_gn=False, adaptive_group_norm=True)
        self.sigmoid = nn.Sigmoid()
        self.attention_type = attention_type

    def forward(self, x):
        conv_x = x
        for layer_name in self.att_block:
            conv_x = getattr(self, layer_name)(conv_x)
        conv_x = self.att_conv(conv_x)
        attention_map = self.sigmoid(self.att_generator(conv_x))
        if self.attention_type == 1:
            x = x * attention_map
        elif self.attention_type == 2:
            x = x + x * attention_map
        else:
            pass

        # bs, cs, ys, xs = x.shape
        # self.attention_map = self.attention_map.view(bs, 1, ys, xs)

        return x, conv_x, attention_map


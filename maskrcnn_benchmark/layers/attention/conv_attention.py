import torch
import torch.nn as nn
from maskrcnn_benchmark.modeling.make_layers import make_blocks


class ConvAttention(nn.Module):

    def __init__(self, in_channels, cfg):
        super(ConvAttention, self).__init__()
        layers = cfg.CONV_LAYERS
        dilation = cfg.DILATION
        use_gn = cfg.USE_GN
        use_dcn = cfg.USE_DCN
        dcn_config = {
            "with_modulated_dcn": cfg.WITH_MODULATED_DCN,
            "deformable_groups": cfg.DEFORMABLE_GROUPS,
        }
        self.attention_blocks, next_feature = make_blocks(layers, in_channels,
                                                          pattern='attention_conv',
                                                          parent=self,
                                                          dilation=dilation,
                                                          use_gn=use_gn,
                                                          use_dcn=use_dcn,
                                                          dcn_config=dcn_config, )

    def forward(self, query, key, value):
        for layer_name in self.attention_blocks:
            query = getattr(self, layer_name)(query)
        for layer_name in self.attention_blocks:
            key = getattr(self, layer_name)(key)
        query_key = query + key
        for layer_name in self.attention_blocks:
            query_key = getattr(self, layer_name)(query_key)
        for layer_name in self.attention_blocks:
            value = getattr(self, layer_name)(value)
        value = value + query_key
        for layer_name in self.attention_blocks:
            value = getattr(self, layer_name)(value)
        return value


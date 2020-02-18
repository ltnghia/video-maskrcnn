import torch.nn as nn
import torch.nn.functional as F
import torch
from maskrcnn_benchmark.layers.non_local import NonLocal2D
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.utils import dfs_freeze


class FPNAttention(nn.Module):
    def __init__(self, use_gamma=False, alpha=0.5):
        super(FPNAttention, self).__init__()
        self.alpha = alpha
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(1))
        else:
            self.gamma = None

    def forward(self, pyramid_features, balance_feature, refine_level=2, interpolation_method=1, attention_map=None):
        output = []

        for i in range(len(pyramid_features)):
            pyramid_feature = pyramid_features[i]
            out_size = pyramid_features[i].size()[2:]

            if attention_map is not None:
                attention = F.interpolate(attention_map, size=out_size, mode='nearest')
                pyramid_feature = pyramid_feature + pyramid_feature * attention

            if balance_feature is not None:
                if interpolation_method == 1:
                    if i < refine_level:
                        residual = F.interpolate(balance_feature, size=out_size, mode='nearest')
                    else:
                        residual = F.adaptive_max_pool2d(balance_feature, output_size=out_size)
                else:
                    if balance_feature.shape[2] < out_size[0] or balance_feature.shape[3] < out_size[1]:
                        residual = F.interpolate(balance_feature, size=out_size, mode='nearest')
                    else:
                        residual = F.adaptive_max_pool2d(balance_feature, output_size=out_size)
                if self.gamma is not None:
                    # print('gamma', self.gamma)
                    pyramid_feature = self.gamma * residual + pyramid_feature
                else:
                    pyramid_feature = self.alpha * residual + (1-self.alpha) * pyramid_feature

            output.append(pyramid_feature)
        output = tuple(output)
        return output
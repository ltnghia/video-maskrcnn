import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers.non_local import NonLocal2D
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.utils import dfs_freeze


class BFP(nn.Module):
    """BFP (Balanced Feature Pyramids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 refine_level=2,
                 refine_type='none',
                 use_gn=False,
                 freeze=False,):
        super(BFP, self).__init__()
        assert refine_type in ['none', 'conv', 'non_local']

        self.in_channels = in_channels
        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level

        if self.refine_type == 'conv':
            self.refine = make_conv3x3(
                self.in_channels,
                self.in_channels,
                use_gn=use_gn,
                use_relu=True,
                kaiming_init=True)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                use_gn=use_gn,)
        else:
            self.refine = None

        self.freeze = freeze
        if self.freeze:
            dfs_freeze(self, requires_grad=False)

    def forward(self, inputs, balance_feature=None, compute_bfp=True, alpha2=1):
        if balance_feature is None:
            # step 1: gather multi-level features by resize and average
            balance_feature = BFP.get_balance_feature(inputs, refine_level=self.refine_level)

            # step 2: refine gathered features
            if self.refine:
                balance_feature = self.refine(balance_feature)

        if not compute_bfp:
            bfp = inputs
        else:
            # step 3: scatter refined features to multi-levels by a residual path
            bfp = BFP.compute_bfp(inputs, balance_feature, refine_level=self.refine_level, interpolation_method=1, alpha2=alpha2)

        return bfp, balance_feature

    @staticmethod
    def get_balance_feature(pyramid_features, refine_level=2):
        feats = []
        gather_size = pyramid_features[refine_level].size()[2:]
        for i in range(len(pyramid_features)):
            if i < refine_level:
                gathered = F.adaptive_max_pool2d(pyramid_features[i], output_size=gather_size)
            else:
                gathered = F.interpolate(pyramid_features[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)
        return bsf

    @staticmethod
    def compute_bfp(pyramid_features, balance_feature, refine_level=2, interpolation_method=1, alpha=0.5, alpha2=1):
        bfp = []

        for i in range(len(pyramid_features)):
            pyramid_feature = pyramid_features[i]
            out_size = pyramid_features[i].size()[2:]

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
                pyramid_feature = alpha * residual + alpha2 * pyramid_feature

            bfp.append(pyramid_feature)
        bfp = tuple(bfp)
        return bfp


def build_bfp(cfg_bfp, in_channels):
    model = BFP(in_channels,
                refine_level=cfg_bfp.REFINE_LEVEL,
                refine_type=cfg_bfp.REFINE_TYPE,
                use_gn=cfg_bfp.USE_GN,
                )
    return model


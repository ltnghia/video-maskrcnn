# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
from torch import nn
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d


def get_group_gn(dim, dim_per_gp, num_groups, adaptive=False):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        if not adaptive:
            assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
            group_gn = dim // dim_per_gp
        else:
            if dim % dim_per_gp == 0:
                group_gn = dim // dim_per_gp
            else:
                group_gn = 1
    else:
        if not adaptive:
            assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
            group_gn = num_groups
        else:
            if dim % num_groups == 0:
                group_gn = num_groups
            else:
                group_gn = 1

    return group_gn


def group_norm(out_channels, affine=True, divisor=1, adaptive=False):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups, adaptive=adaptive),
        out_channels,
        eps,
        affine
    )


def make_conv1x1(
    in_channels,
    out_channels,
    use_gn=False,
    use_relu=False,
    use_bias=True,
    kaiming_init=True,
    adaptive_group_norm=False,
):
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        bias=False if not use_bias or use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn and use_bias:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels, adaptive=adaptive_group_norm))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_gn=False,
    use_relu=False,
    use_bias=True,
    kaiming_init=True,
    adaptive_group_norm=False,
):
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if not use_bias or use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn and use_bias:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels, adaptive=adaptive_group_norm))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(dim_in, hidden_dim, use_gn=False, use_bias=True, kaiming_init=False, adaptive_group_norm=False,):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''

    fc = nn.Linear(dim_in, hidden_dim, bias=False if not use_bias or use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
    else:
        nn.init.kaiming_uniform_(fc.weight, a=1)

    if use_bias and not use_gn:
        nn.init.constant_(fc.bias, 0)

    if use_gn:
        return nn.Sequential(fc, group_norm(hidden_dim, adaptive=adaptive_group_norm))

    # if use_gn:
    #     fc = nn.Linear(dim_in, hidden_dim, bias=False)
    #     if kaiming_init:
    #         nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
    #     else:
    #         nn.init.kaiming_uniform_(fc.weight, a=1)
    #     return nn.Sequential(fc, group_norm(hidden_dim))
    # fc = nn.Linear(dim_in, hidden_dim)
    # if kaiming_init:
    #     nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
    # else:
    #     nn.init.kaiming_uniform_(fc.weight, a=1)
    # nn.init.constant_(fc.bias, 0)

    return fc


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


def make_blocks(layers, input_size,
                pattern='',
                parent=None,
                dilation=1,
                use_gn=True,
                use_dcn=False,
                dcn_config={}, ):
    next_feature = input_size
    blocks = []
    for layer_idx, layer_features in enumerate(layers, 1):
        layer_name = pattern + "{}".format(layer_idx)
        if use_dcn:
            module = DFConv2d(
                next_feature,
                layer_features,
                with_modulated_dcn=dcn_config.get("with_modulated_dcn", False),
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=dilation,
                deformable_groups=dcn_config.get("deformable_groups", 1),
                bias=False if use_gn else True,
            )
            module = [module, ]
            if use_gn:
                module.append(group_norm(layer_features))
            module.append(nn.ReLU(inplace=True))
            if len(module) > 1:
                module = nn.Sequential(*module)
        else:
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn, use_relu=True
            )
        if parent is not None:
            parent.add_module(layer_name, module)
        next_feature = layer_features
        blocks.append(layer_name)
    return blocks, next_feature

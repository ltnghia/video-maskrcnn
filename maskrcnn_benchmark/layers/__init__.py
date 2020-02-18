# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .misc import Conv2d, ConvTranspose2d, DFConv2d, BatchNorm2d, interpolate, FrozenBatchNorm2d
from .nms import nms

from maskrcnn_benchmark.layers.pooling.roi_pool import ROIPool, roi_pool
from maskrcnn_benchmark.layers.pooling.roi_align import roi_align, ROIAlign
from maskrcnn_benchmark.layers.pooling.PreciseRoIPooling.prroi_pool import PrRoIPool2D

from maskrcnn_benchmark.layers.loss.sigmoid_focal_loss import SigmoidFocalLoss
from maskrcnn_benchmark.layers.loss.cross_entropy_loss import cross_entropy, binary_cross_entropy, weighted_binary_cross_entropy, mask_cross_entropy
from maskrcnn_benchmark.layers.loss.class_balance_loss import ClassBalanceLoss
from maskrcnn_benchmark.layers.loss.smooth_l1_loss import SmoothL1Loss
from maskrcnn_benchmark.layers.loss.adjust_smooth_l1_loss import AdjustSmoothL1Loss
from maskrcnn_benchmark.layers.loss.balanced_l1_loss import BalancedL1Loss
from maskrcnn_benchmark.layers.loss.l2_loss import l2_loss
from maskrcnn_benchmark.layers.loss.wing_loss import WingLoss
from maskrcnn_benchmark.layers.loss.iou_loss import IOULoss

from .dcn.deform_conv_func import deform_conv, modulated_deform_conv
from .dcn.deform_conv_module import DeformConv, ModulatedDeformConv, ModulatedDeformConvPack
from .dcn.deform_pool_func import deform_roi_pooling
from .dcn.deform_pool_module import DeformRoIPooling, DeformRoIPoolingPack, ModulatedDeformRoIPoolingPack

__all__ = [ "nms",
            "roi_align",
            "ROIAlign",
            "roi_pool",
            "ROIPool",
            "PrRoIPool2D",
            "Conv2d",
            "DFConv2d",
            "ConvTranspose2d",
            "interpolate",
            "BatchNorm2d",
            "FrozenBatchNorm2d",
            "cross_entropy",
            "binary_cross_entropy",
            "mask_cross_entropy",
            "weighted_binary_cross_entropy",
            "SigmoidFocalLoss",
            "ClassBalanceLoss",
            "SmoothL1Loss",
            "AdjustSmoothL1Loss",
            "BalancedL1Loss",
            "l2_loss",
            "WingLoss",
            "IOULoss",
            'deform_conv',
            'modulated_deform_conv',
            'DeformConv',
            'ModulatedDeformConv',
            'ModulatedDeformConvPack',
            'deform_roi_pooling',
            'DeformRoIPooling',
            'DeformRoIPoolingPack',
            'ModulatedDeformRoIPoolingPack',
          ]


#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : prroi_pool.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
# 
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import torch.nn as nn

from .functional import prroi_pool2d

__all__ = ['PrRoIPool2D']


class PrRoIPool2D(nn.Module):
    def __init__(self, output_size, spatial_scale, use_torchvision=False):
        super().__init__()

        self.pooled_height = int(output_size[0])
        self.pooled_width = int(output_size[1])
        self.spatial_scale = float(spatial_scale)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            print('torch vision has not supported PrRoIPool2D')
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)

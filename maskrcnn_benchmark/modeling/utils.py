# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def dfs_freeze(model, requires_grad=False, display=False):
    if model is None:
        return
    for name, child in model.named_children():
        for param in child.parameters():
            if display:
                print(name, param.requires_grad, requires_grad)
            param.requires_grad = requires_grad
        dfs_freeze(child)

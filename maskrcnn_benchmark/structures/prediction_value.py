import cv2
import copy
import torch
import numpy as np
from maskrcnn_benchmark.layers.misc import interpolate


class PredictionValueList(object):

    def __init__(self, value):

        if isinstance(value, torch.Tensor):
            # The raw data representation is passed as argument
            value = value.clone()
        elif isinstance(value, PredictionValueList):
            # just hard copy the BinaryMaskList instance's underlying data
            value = value.value.clone()
        else:
            RuntimeError("Type of `value` argument could not be interpreted:%s" % type(value))

        self.value = value

    def to(self, device):
        self.value = self.value.to(device)
        return self

    def get_tensors(self):
        return self.value

    def cat(self, x, dim=0):
        if isinstance(x, PredictionValueList):
            value = x.get_tensors()
        self.value = torch.cat((self.get_tensors(), value), dim=dim)

    def to_list(self):
        results = []
        value = self.get_tensors()
        for i in range(value.shape[0]):
            results.append(PredictionValueList(value[i]))
        return results

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "prediction_value={}, ".format(self.value.cpu().detach().numpy())
        return s

    @staticmethod
    def to_device(x, device):
        if x is None:
            return None
        if isinstance(x, (tuple, list)):
            if x[0] is None:
                return None
            output = PredictionValueList(x[0])
            output.value = [xi.value for xi in x]
            output.value = torch.stack(output.value)
            output = output.to(device)
        else:
            output = x.to(device)
        return output.value


def to_value_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, PredictionValueList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        return PredictionValueList(tensors)
    elif isinstance(tensors, (tuple, list)):
        batch_shape = (len(tensors),)
        batch_values = tensors[0].new(*batch_shape).zero_()
        for i, x in enumerate(tensors):
            batch_values[i] = x
        return PredictionValueList(batch_values)
    else:
        raise TypeError("Unsupported type for to_value_list: {}".format(type(tensors)))


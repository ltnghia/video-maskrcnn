import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class GlobalAveragePool(nn.Module):

    def __init__(self):
        super(GlobalAveragePool, self).__init__()

    def forward(self, x, transform=True):
        # x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        if transform:
            x = x.view(x.size(0), -1)
        return x



# RetinaMask: Learning to predict masks improves state-of-the-art single-shot detection for free, 2019

import torch
from torch import nn
from .utils import add_weights_and_normalize


class AdjustSmoothL1Loss(nn.Module):
    def __init__(self, num_features, momentum=0.1, beta=1. / 9):
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer('running_mean', torch.empty(num_features).fill_(beta))
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, inputs, target, weight=None,  reduction='mean'):
        n = torch.abs(inputs -target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n.mean(dim=0))
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n.var(dim=0))

        beta = (self.running_mean - self.running_var)
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return add_weights_and_normalize(loss,
                                         label=target,
                                         weight=weight,
                                         reduction=reduction)

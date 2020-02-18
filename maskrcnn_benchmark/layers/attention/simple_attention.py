import torch
import torch.nn as nn


class SimpleAttention(nn.Module):

    def __init__(self, use_gamma=False, alpha=0.5):
        super(SimpleAttention, self).__init__()
        self.alpha = alpha
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(1))
        else:
            self.gamma = None

    def forward(self, key, value):
        if self.gamma is not None:
            x = self.gamma * key + value
        else:
            x = self.alpha * value + (1-self.alpha) * key
        return x




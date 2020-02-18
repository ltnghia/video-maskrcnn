import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """ Feature Attention Layer"""

    def __init__(self, in_channels, activation='relu', use_gamma=True):
        super(FeatureAttention, self).__init__()
        self.in_channels = in_channels
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)

        self.query_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        if use_gamma:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = None

    def forward(self, query, key, value, x=None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        if query is None:
            key = query
        elif key is None:
            query = key

        proj_query = self.query_conv(query)
        proj_key = self.key_conv(key)
        proj_value = self.value_conv(value)

        proj_query = proj_query.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = proj_key.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        proj_value = proj_value.view(m_batchsize, -1, width * height)  # B X C X N

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        if self.gamma is not None and x is not None:
            out = self.gamma * out + x
            # print(self.gamma)
        return out, attention
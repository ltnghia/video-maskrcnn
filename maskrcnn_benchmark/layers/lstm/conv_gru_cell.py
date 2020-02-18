import torch
from torch import nn
# from torch.autograd import Variable
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, bias=1, device=torch.device("cuda"), phase='train'):
        super(ConvGRUCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                                   padding=self.padding)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.padding)
        self.phase = phase

    def forward(self, input, pre_state):
        if pre_state is None:
            size_h = [input.size()[0], self.hidden_size] + list(input.size()[2:])
            pre_state = (torch.zeros(size_h, requires_grad=(True, False)[self.phase == 'test0']).to(self.device),)

        hidden = pre_state[-1]
        c1 = self.ConvGates(torch.cat((F.dropout(input, p=0.2, training=(False, True)[self.phase=='train']), hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = torch.sigmoid(rt)
        update_gate = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = torch.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return (next_h, )

    def init_state(self, input):
        size_h = [input.size()[0], self.hidden_size] + list(input.size()[2:])
        state = torch.zeros(size_h, requires_grad=(True, False)[self.phase == 'test0']).to(self.device),
        return state
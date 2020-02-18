import torch
from torch import nn
# from torch.autograd import Variable
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional lstm cell
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=1, device=torch.device("cuda"), phase='train'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.phase = phase
        self.gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, 
                               kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_tensor.data.size()[0]
        spatial_size = input_tensor.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_hidden(batch_size, spatial_size)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        if self.phase == 'train':
            stacked_inputs = torch.cat(
                (F.dropout(input_tensor, p=0.2, training=(False, True)[self.phase == 'train']), prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((input_tensor, prev_hidden), 1)
        gates = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size, input_size):
        return (torch.zeros(batch_size, self.hidden_dim, input_size[0], input_size[1],
                            requires_grad=(True, False)[self.phase == 'test0']).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, input_size[0], input_size[1],
                            requires_grad=(True, False)[self.phase == 'test0']).to(self.device))

    def init_state(self, input_tensor):
        batch_size = input_tensor.size()[0]
        spatial_size = input_tensor.size()[2:]
        state_size = [batch_size, self.hidden_dim] + list(spatial_size)
        state = (
            torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test0']).to(self.device),
            torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test0']).to(self.device),
        )

        return state


def _main():
    """
    Run some basic tests on the API
    """

    KERNEL_SIZE = 3

    # define batch_size, channels, height, width
    b, c, h, w = 1, 3, 4, 8
    d = 5           # hidden state size
    lr = 1e-1       # learning rate
    T = 6           # sequence length
    max_epoch = 20  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    device = torch.device("cuda")

    print('Instantiate model')
    model = ConvLSTMCell(c, d, KERNEL_SIZE, 1, device)
    print(repr(model))

    model.to(device)

    print('Create input and target Variables')
    x = torch.rand(T, b, c, h, w).to(device)
    y = torch.randn(T, b, d, h, w).to(device)
    # x = Variable(torch.rand(T, b, c, h, w)).to(device)
    # y = Variable(torch.randn(T, b, d, h, w)).to(device)

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        for t in range(0, T):
            state = model(x[t], state)
            loss += loss_fn(state[0], y[t])

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data.item()))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    _main()


__author__ = "Alfredo Canziani"
__credits__ = ["Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Prototype"  # "Prototype", "Development", or "Production"
__date__ = "Jan 17"

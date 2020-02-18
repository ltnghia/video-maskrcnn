from maskrcnn_benchmark.layers.lstm.conv_lstm_cell import ConvLSTMCell

import torch
from torch import nn


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size,
                 batch_first=True, bias=True, return_all_layers=False, reverse=False, device=torch.device("cuda")):
        super(ConvLSTM, self).__init__()

        num_layers = len(hidden_dim)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.reverse = reverse
        self.device = device

        cell_list = []
        self.name_list = []
        for i in range(0, self.num_layers):
            name = 'ConvLSTMCell_{}'.format(i)
            self.name_list.append(name)
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell = ConvLSTMCell(cur_input_dim, self.hidden_dim[i], kernel_size=self.kernel_size[i], bias=self.bias, device=self.device)
            setattr(self, name, cell)
            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), input_size=[input_tensor.size(3), input_tensor.size(4)])

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        layer_output_list = []
        last_state_list = []

        for layer_idx in range(self.num_layers):
            # name = self.name_list[layer_idx]
            # cell_tmp = getattr(self, name)
            cell = self.cell_list[layer_idx]
            h, c = hidden_state[layer_idx]
            output_inner = [None] * seq_len

            seq_id = range(seq_len)
            if self.reverse:
                seq_id = reversed(seq_id)

            for t in seq_id:
                h, c = cell(input_tensor=cur_layer_input[:, t, :, :, :], prev_state=[h, c])
                output_inner[t] = h

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, input_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, input_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    b, c, h, w = 1, 3, 4, 8
    d = 5  # output state size
    lr = 0.5  # learning rate
    T = 6  # sequence length
    max_epoch = 1000  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    device = torch.device("cuda")

    print('Instantiate model')
    model = ConvLSTM(input_dim=c,
                 hidden_dim=[d],
                 kernel_size=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 reverse=True,
                 device=device)
    print(repr(model))

    model.to(device)

    print('Create input and target Variables')
    input = torch.rand(b, T, c, h, w).to(device)
    target = torch.randn(b, T, d, h, w).to(device)
    # input = Variable(torch.rand(b, T, c, h, w)).to(device)
    # target = Variable(torch.randn(b, T, d, h, w)).to(device)

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    # gradient check
    output = model(input)
    output = output[0][0]
    res = torch.autograd.gradcheck(loss_fn, (output.double(), target.double()), eps=1e-6, raise_exception=True)
    print('gradient check', res)

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        epoch += 1
        state = None
        loss = 0
        if not epoch%500:
            lr /= 10

        output = model(input)
        output = output[0][0]
        loss += loss_fn(output, target)

        print(' > Epoch {:2d} lr {} loss: {:.3f}'.format((epoch + 1), lr, loss.data.item()))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', input.shape)
    print('Target size:', output.shape)

if __name__ == '__main__':
    _main()

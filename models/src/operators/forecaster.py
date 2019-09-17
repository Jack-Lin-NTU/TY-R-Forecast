import torch
from torch import nn
from src.utils.utils import make_layers

class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, seq_len):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self.seq_len = seq_len

        for index, (subnet, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), subnet)

    def forward_by_stage(self, input_, state, subnet, rnn):
        input_, state_stage = rnn(input_, state, self.seq_len)
        batch_size, seq_number, input_channel, height, width = input_.size()
        input_ = torch.reshape(input_, (-1, input_channel, height, width))
        input_ = subnet(input_)
        input_ = torch.reshape(input_, (batch_size, seq_number, input_.shape[1], input_.size(2), input_.size(3)))

        return input_

        # input_: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input_ = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'), getattr(self, 'rnn3'))        
        for i in list(range(1, self.blocks))[::-1]:
            input_ = self.forward_by_stage(input_, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input_
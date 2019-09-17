import torch
from torch import  nn
import torch.nn.functional as F
from src.utils.utils import make_layers

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (subnet, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), subnet)
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input_, subnet, rnn):
        batch_size, seq_number, input_channel, height, width = input_.shape
        input_ = torch.reshape(input_, (-1, input_channel, height, width))
        input_ = subnet(input_)
        input_ = torch.reshape(input_, (batch_size, seq_number, input_.shape[1], input_.shape[2], input_.shape[3]))
        outputs_stage, state_stage = rnn(input_, None)

        return outputs_stage, state_stage

    # input_: 5D S*B*I*H*W
    def forward(self, input_):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input_, state_stage = self.forward_by_stage(input_, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

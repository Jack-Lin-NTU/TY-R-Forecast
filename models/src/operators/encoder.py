import torch
from torch import  nn
import torch.nn.functional as F
from src.utils.utils import make_layers

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input_, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input_.shape
        input_ = torch.reshape(input_, (-1, input_channel, height, width))
        input_ = subnet(input_)
        input_ = torch.reshape(input_, (seq_number, batch_size, input_.size(1), input_.size(2), input_.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input_.size(3), input_.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, input_.size(3), input_.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(input_, None)

        return outputs_stage, state_stage

    # input_: 5D S*B*I*H*W
    def forward(self, input_):
        hidden_states = []
        logging.debug(input_.size())
        for i in range(1, self.blocks+1):
            input_, state_stage = self.forward_by_stage(input_, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

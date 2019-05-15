from easydict import EasyDict as edict

def MYMODEL_HYPERPARAMs(args):
    c = args.channel_factor
    MYMODEL = edict({
                    'input_frames': args.input_frames,
                    'target_frames': args.target_frames,
                    'TyCatcher_channel_input': 8,
                    'TyCatcher_channel_hidden': [20,6],
                    'TyCatcher_channel_n_layers': 2,
                    'gru_channel_input': args.input_channels ,
                    'gru_channel_hidden':1,
                    'gru_kernel': 3,
                    'gru_stride': 1,
                    'gru_padding': 1,
                    })

    return MYMODEL
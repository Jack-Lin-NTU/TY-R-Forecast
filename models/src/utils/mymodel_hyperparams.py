from easydict import EasyDict as edict

def MYSINGLEMODEL_HYPERPARAMs(args):
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

def MYMULTIMODEL_HYPERPARAMs(args):
    c = args.channel_factor
    MYMODEL = edict({
                    'input_frames': args.input_frames,
                    'target_frames': args.target_frames,
                    # hyperparameters for TY Catcher
                    'TyCatcher_input': 8,
                    'TyCatcher_hidden': [20,6],
                    'TyCatcher_n_layers': 2,
                    # hyperparameters for the Encoder
                    'encoder_input': args.input_channels,
                    'encoder_downsample': [4*c, 32*c, 96*c],
                    'encoder_gru': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [7,4,3],
                    'encoder_downsample_s': [5,3,2],
                    'encoder_downsample_p': 1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    # hyperparameters for the Forecaster
                    'forecaster_upsample_cin': [96*c,96*c,32*c],
                    'forecaster_upsample_cout': [96*c,96*c,4*c],
                    'forecaster_upsample_k': [3,4,7],
                    'forecaster_upsample_s': [2,3,5],
                    'forecaster_upsample_p': 1,
                    'forecaster_n_layers': 3,
                    # hyperparameters for output layer in the Forecaster
                    'forecaster_output_cout': 1,
                    'forecaster_output_k': 3,
                    'forecaster_output_s': 1,
                    'forecaster_output_p': 1,
                    'forecaster_n_output_layers': 1,
                    })
    return MYMODEL

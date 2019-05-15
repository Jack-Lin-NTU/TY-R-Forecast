from easydict import EasyDict as edict

def TRAJGRU_HYPERPARAMs(args):
    c = args.channel_factor
    TRAJGRU = edict({
                    'channel_factor': c,
                    'rnn_link_size': [13, 13, 9],
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [4*c, 32*c, 96*c],
                    'encoder_rnn_channels': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [7,4,3],
                    'encoder_downsample_s': [5,3,2],
                    'encoder_downsample_p':[1,1,1],
                    'encoder_rnn_k': [3,3,3],
                    'encoder_rnn_s': [1,1,1],
                    'encoder_rnn_p': [1,1,1],
                    'encoder_n_layers': 6,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [96*c,96*c,4*c],
                    'forecaster_rnn_channels': [96*c,96*c,32*c],
                    'forecaster_upsample_k': [3,4,7],
                    'forecaster_upsample_s': [2,3,5],
                    'forecaster_upsample_p': [1,1,1],
                    'forecaster_rnn_k': [3,3,3],
                    'forecaster_rnn_s': [1,1,1],
                    'forecaster_rnn_p': [1,1,1],
                    'forecaster_n_layers': 6,
                    'forecaster_output_channels': 1,
                    'forecaster_output_k': 3,
                    'forecaster_output_s': 1,
                    'forecaster_output_p': 1,
                    'forecaster_output_layers': 1,
                    })

    return TRAJGRU

def CONVGRU_HYPERPARAMs(args):
    c = args.channel_factor
    CONVGRU = edict({
                    'channel_factor': c,
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [4*c, 32*c, 96*c],
                    'encoder_rnn_channels': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [7,4,3],
                    'encoder_downsample_s': [5,3,2],
                    'encoder_downsample_p':[1,1,1],
                    'encoder_rnn_k': [3,3,3],
                    'encoder_rnn_s': [1,1,1],
                    'encoder_rnn_p': [1,1,1],
                    'encoder_n_layers': 6,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [96*c,96*c,4*c],
                    'forecaster_rnn_channels': [96*c,96*c,32*c],
                    'forecaster_upsample_k': [3,4,7],
                    'forecaster_upsample_s': [2,3,5],
                    'forecaster_upsample_p': [1,1,1],
                    'forecaster_rnn_k': [3,3,3],
                    'forecaster_rnn_s': [1,1,1],
                    'forecaster_rnn_p': [1,1,1],
                    'forecaster_n_layers': 6,
                    'forecaster_output_channels': 1,
                    'forecaster_output_k': 3,
                    'forecaster_output_s': 1,
                    'forecaster_output_p': 1,
                    'forecaster_output_layers': 1,
                    })

    return CONVGRU
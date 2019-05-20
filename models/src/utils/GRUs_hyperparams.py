from easydict import EasyDict as edict

def TRAJGRU_HYPERPARAMs(args):
    c = args.channel_factor
    TRAJGRU = edict({
                    'channel_factor': c,
                    'gru_link_size': [13, 13, 9],
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [4*c, 32*c, 96*c],
                    'encoder_gru_channels': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [7,4,3],
                    'encoder_downsample_s': [5,3,2],
                    'encoder_downsample_p':1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [96*c, 96*c, 4*c],
                    'forecaster_gru_channels': [96*c, 96*c, 32*c],
                    'forecaster_upsample_k': [3,4,7],
                    'forecaster_upsample_s': [2,3,5],
                    'forecaster_upsample_p': 1,
                    'forecaster_gru_k': 3,
                    'forecaster_gru_s': 1,
                    'forecaster_gru_p': 1,
                    'forecaster_n_cells': 3,
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
                    'encoder_gru_channels': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [7,4,3],
                    'encoder_downsample_s': [5,3,2],
                    'encoder_downsample_p': 1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [96*c, 96*c, 4*c],
                    'forecaster_gru_channels': [96*c, 96*c, 32*c],
                    'forecaster_upsample_k': [3,4,7],
                    'forecaster_upsample_s': [2,3,5],
                    'forecaster_upsample_p': 1,
                    'forecaster_gru_k': 3,
                    'forecaster_gru_s': 1,
                    'forecaster_gru_p': 1,
                    'forecaster_n_cells': 3,
                    'forecaster_output_channels': 1,
                    'forecaster_output_k': 3,
                    'forecaster_output_s': 1,
                    'forecaster_output_p': 1,
                    'forecaster_output_layers': 1,
                    })

    return CONVGRU
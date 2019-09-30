from easydict import EasyDict as edict

def TRAJGRU_HYPERPARAMs(args):
    c = args.channel_factor
    TRAJGRU = edict({
                    'n_encoders': args.I_nframes,
                    'n_forecasters': args.F_nframes,
                    'channel_factor': c,
                    'gru_link_size': [13, 13, 9],
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [4*c, 32*c, 96*c],
                    'encoder_gru_channels': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [5,4,4],
                    'encoder_downsample_s': [3,2,2],
                    'encoder_downsample_p':1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [96*c, 96*c, 4*c],
                    'forecaster_gru_channels': [96*c, 96*c, 32*c],
                    'forecaster_upsample_k': [4,4,5],
                    'forecaster_upsample_s': [2,2,3],
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

def DEC_TRAJGRU_HYPERPARAMs(args):
    c = args.channel_factor
    TRAJGRU = edict({
                    'n_encoders': args.I_nframes,
                    'n_forecasters': args.F_nframes,
                    'channel_factor': c,
                    'gru_link_size': [13, 13, 9],
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [64*c, 32*c, 16*c],
                    'encoder_gru_channels': [32*c, 16*c, 16*c],
                    'encoder_downsample_k': [5,4,4],
                    'encoder_downsample_s': [3,2,2],
                    'encoder_downsample_p':1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [16*c, 16*c, 64*c],
                    'forecaster_gru_channels': [16*c, 16*c, 32*c],
                    'forecaster_upsample_k': [4,4,5],
                    'forecaster_upsample_s': [2,2,3],
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
                    'n_encoders': args.I_nframes,
                    'n_forecasters': args.F_nframes,
                    'channel_factor': c,
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [4*c, 32*c, 96*c],
                    'encoder_gru_channels': [32*c, 96*c, 96*c],
                    'encoder_downsample_k': [5,4,3],
                    'encoder_downsample_s': [3,3,2],
                    'encoder_downsample_p': 1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [96*c, 96*c, 4*c],
                    'forecaster_gru_channels': [96*c, 96*c, 32*c],
                    'forecaster_upsample_k': [3,4,5],
                    'forecaster_upsample_s': [2,3,3],
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

def DEC_CONVGRU_HYPERPARAMs(args):
    c = args.channel_factor
    CONVGRU = edict({
                    'n_encoders': args.I_nframes,
                    'n_forecasters': args.F_nframes,
                    'channel_factor': c,
                    'encoder_input_channel': args.input_channels,
                    'encoder_downsample_channels': [64*c, 32*c, 16*c],
                    'encoder_gru_channels': [32*c, 16*c, 16*c],
                    'encoder_downsample_k': [5,4,3],
                    'encoder_downsample_s': [3,3,2],
                    'encoder_downsample_p': 1,
                    'encoder_gru_k': 3,
                    'encoder_gru_s': 1,
                    'encoder_gru_p': 1,
                    'encoder_n_cells': 3,
                    'forecaster_input_channel': 0,
                    'forecaster_upsample_channels': [16*c, 16*c, 64*c],
                    'forecaster_gru_channels': [16*c, 16*c, 32*c],
                    'forecaster_upsample_k': [3,4,5],
                    'forecaster_upsample_s': [2,3,3],
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


# def MYMODEL_HYPERPARAMs(args):
#     c = args.channel_factor
#     if args.catcher_location:
#         TyCatcher_input = 3
#     else:
#         TyCatcher_input = 8
#     MYMODEL = edict({
#                     'input_frames': args.input_frames,
#                     'target_frames': args.target_frames,
#                     # hyperparameters for TY Catcher
#                     'TyCatcher_input': TyCatcher_input,
#                     'TyCatcher_hidden': [20,6],
#                     'TyCatcher_n_layers': 2,
#                     # hyperparameters for the Encoder
#                     'encoder_input': args.input_channels,
#                     'encoder_downsample': [4*c, 32*c, 96*c],
#                     'encoder_gru': [32*c, 96*c, 96*c],
#                     'encoder_downsample_k': [7,5,4],
#                     'encoder_downsample_s': [5,3,2],
#                     'encoder_downsample_p': 1,
#                     'encoder_gru_k': 3,
#                     'encoder_gru_s': 1,
#                     'encoder_gru_p': 1,
#                     'encoder_n_cells': 3,
#                     # hyperparameters for the Forecaster
#                     'forecaster_upsample_cin': [96*c,96*c,32*c],
#                     'forecaster_upsample_cout': [96*c,96*c,4*c],
#                     'forecaster_upsample_k': [4,5,7],
#                     'forecaster_upsample_s': [2,3,5],
#                     'forecaster_upsample_p': 1,
#                     'forecaster_n_layers': 3,
#                     # hyperparameters for output layer in the Forecaster
#                     'forecaster_output_cout': 1,
#                     'forecaster_output_k': 3,
#                     'forecaster_output_s': 1,
#                     'forecaster_output_p': 1,
#                     'forecaster_n_output_layers': 1,
#                     })
#     return MYMODEL

# def TEST_HYPERPARAMs(args):
#     c = args.channel_factor
#     TRAJGRU = edict({
#                     'channel_factor': c,
#                     'gru_link_size': [13],
#                     'encoder_input_channel': args.input_channels,
#                     'encoder_downsample_channels': [4*c],
#                     'encoder_gru_channels': [32*c],
#                     'encoder_downsample_k': [7],
#                     'encoder_downsample_s': [5],
#                     'encoder_downsample_p':1,
#                     'encoder_gru_k': 3,
#                     'encoder_gru_s': 1,
#                     'encoder_gru_p': 1,
#                     'encoder_n_cells': 1,
#                     # hyperparameters for the Forecaster
#                     'forecaster_u_cin': [96*c,96*c,32*c],
#                     'forecaster_upsample_cout': [96*c,96*c,4*c],
#                     'forecaster_upsample_k': [4,5,7],
#                     'forecaster_upsample_s': [2,3,5],
#                     'forecaster_upsample_p': 1,
#                     'forecaster_n_layers': 3,
#                     # hyperparameters for output layer in the Forecaster
#                     'forecaster_output_cout': 1,
#                     'forecaster_output_k': 3,
#                     'forecaster_output_s': 1,
#                     'forecaster_output_p': 1,
#                     'forecaster_n_output_layers': 1,
#                     })

#     return TRAJGRU

# def FLOWGRU_HYPERPARAMs(args):
#     c = args.channel_factor
#     PARAMs = edict({
#                     'channel_factor': c,
#                     'encoder_cin': 3,
#                     'encoder_d_c': [4*c, 32*c, 96*c],
#                     'encoder_gru_c': [32*c, 96*c, 96*c],
#                     'encoder_d_k': [7,5,4],
#                     'encoder_d_s': [5,3,2],
#                     'encoder_d_p': 1,
#                     'encoder_gru_k': 3,
#                     'encoder_gru_s': 1,
#                     'encoder_gru_p': 1,
#                     # hyperparameters for the Forecaster
#                     'forecaster_u_cin': [96*c,96*c,32*c],
#                     'forecaster_u_cout': [96*c,96*c,4*c],
#                     'forecaster_u_k': [4,5,7],
#                     'forecaster_u_s': [2,3,5],
#                     'forecaster_u_p': 1,
#                     # hyperparameters for output layer in the Forecaster
#                     'c_out': 1,
#                     })

#     return PARAMs
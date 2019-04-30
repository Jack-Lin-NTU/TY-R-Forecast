# import modules
import os
import math
import torch
import pandas as pd
import argparse
import torch.optim as optim
from torch.optim import Optimizer

# This version of Adam keeps an fp32 copy of the paramargseters and 
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the 
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params such that it 
# moves them to gpu 0--if you're using a different GPU or want to 
# do multi-GPU you may need to deal with this.

class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, args=None):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.to(device=args.device, dtype=torch.float32) for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, fp32_p in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss

class loss_rainfall():
    def __init__(self, args):
        self.args = args

    def bmse(self, outputs, labels):
        loss = 0
        outputs_size = outputs.shape[1]
        if args.normalize_target:
            value_list = [0, 2/args.max_values['QPE'], 5/args.max_values['QPE'], 10/args.max_values['QPE'], 30/args.max_values['QPE'], args.max_values['QPE']/args.max_values['QPE']]
            weights = [1, 2, 5, 10, 30]
        else:
            value_list = [0, 2, 5, 10, 30, args.max_values['QPE']]
            weights = [1, 2, 5, 10, 30]
        for i in range(len(value_list)-1):
            mask = torch.stack([value_list[i] <= labels, labels < value_list[i+1]]).all(dim=0)
            loss += torch.sum(weights[i] * ((outputs[mask]-labels[mask])/outputs_size)**2)
        return loss

    def bmae(self, outputs, labels):
        loss = 0
        outputs_size = outputs.shape[1]    
        if args.normalize_target:
            value_list = [0, 2/args.max_values['QPE'], 5/args.max_values['QPE'], 10/args.max_values['QPE'], 30/args.max_values['QPE'], args.max_values['QPE']/args.max_values['QPE']]
            weights = [1, 2, 5, 10, 30]
        else:
            value_list = [0, 2, 5, 10, 30, args.max_values['QPE']]
            weights = [1, 2, 5, 10, 30]
        for i in range(len(value_list)-1):
            mask = torch.stack([value_list[i] <= labels, labels < value_list[i+1]]).all(dim=0)
            loss += torch.sum(weights[i] * torch.abs((outputs[mask]-labels[mask])/outputs_size))

        return loss

def createfolder(directory):
    '''
    This function is used to create new folder with given directory.
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)

def make_path(path, workfolder=None):
    '''
    This function is used to transform path to absolute path.
    '''
    if path[0] == '~':
        new_path = os.path.expanduser(path)
    else:
        new_path = path

    if workfolder is not None and not os.path.isabs(new_path):
        return os.path.join(workfolder, new_path)
        
    return new_path

def remove_file(file):
    if os.path.exists(file):
        os.remove(file)
        
def print_dict(d):
    for key, value in d.items():
        print('{}: {}'.format(key, value))


parser = argparse.ArgumentParser()

working_folder = os.path.expanduser('~/ssd/01_ty_research')
radar_folder = make_path('01_radar_data', working_folder)
weather_folder = make_path('02_weather_data', working_folder)
ty_info_folder =  make_path('03_ty_info', working_folder)

parser.add_argument('--working-folder', metavar='', type=str, default=working_folder,
                   help='The path of working folder.')
parser.add_argument('--radar-folder', metavar='', type=str, default=radar_folder,
                   help='The path of radar folder.')
parser.add_argument('--radar-wrangled-data-folder', metavar='', type=str, default=make_path('02_wrangled_files', radar_folder),
                   help='The path of radar wrangled-data folder.')

parser.add_argument('--weather-folder', metavar='', type=str, default=make_path('02_weather_data', working_folder),
                   help='The path of weather folder.')
parser.add_argument('--weather-wrangled-data-folder', metavar='', type=str, default=make_path('01_wrangled_files', weather_folder),
                   help='The path of weather wrangled-data folder.')

parser.add_argument('--ty-info-folder', metavar='', type=str, default=make_path('03_ty_info', working_folder),
                   help='The path of ty-info folder.')
parser.add_argument('--ty-info-wrangled-data-folder', metavar='', type=str, default=make_path('01_wrangled_files', ty_info_folder),
                   help='The path of ty-info wrangled-data folder.')


parser.add_argument('--result-folder', metavar='', type=str, default=make_path('04_results', working_folder),
                   help='The path of result folder.')
parser.add_argument('--params-folder', metavar='', type=str, default=make_path('05_params', working_folder),
                   help='The path of params folder.')

parser.add_argument('--ty-list', metavar='', type=str, default=make_path('ty_list.csv', working_folder),
                   help='The path of ty list file.')

parser.add_argument('--load-all-data', action='store_true', help='Load all data.')
parser.add_argument('--able-cuda', action='store_true', help='Able cuda.')
parser.add_argument('--gpu', metavar='', type=int, default=0, help='GPU device.(default: 0)')
parser.add_argument('--dtype', metavar='', type=str, default='float32', help='The dtype of values.(default: float32)')

# hyperparameters for training
parser.add_argument('--max-epochs', metavar='', type=int, default=30, help='Max epochs.(default: 30)')
parser.add_argument('--batch-size', metavar='', type=int, default=4, help='Batch size.(default: 8)')
parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='Max epochs.(default: 1e-4)')
parser.add_argument('--lr-scheduler', action='store_true', help='Set lr-scheduler.')
parser.add_argument('--weight-decay', metavar='', type=float, default=0, help='Wegiht decay.(default: 0)')
parser.add_argument('--clip', action='store_true', help='Clip model weightings.')
parser.add_argument('--clip-max-norm', metavar='', type=int, default=300, help='Max norm value for clip model weightings. (default: 300)')
parser.add_argument('--batch-norm', action='store_true', help='Do batch normalization.')

parser.add_argument('--normalize-target', action='store_true', help='Normalize target maps.')

parser.add_argument('--loss-function', metavar='', type=str, default='BMSE', help='The loss function.(default: BMSE)')
parser.add_argument('--input-frames', metavar='', type=int, default=6, help='The size of input frames. (default: 6)')
parser.add_argument('--input-with-grid', action='store_true', help='Input with grid data.')
parser.add_argument('--input-with-QPE', action='store_true', help='Input with QPE data.')
parser.add_argument('--target-frames', metavar='', type=int, default=18, help='The size of target frames. (default: 18)')
parser.add_argument('--channel-factor', metavar='', type=int, default=2, help='Channel factor. (default: 2)')

# parser.add_argument('--I-x-l', metavar='', type=float, default=120.9625, help='The lowest longitude of input map. (default: 120.9625)')
# parser.add_argument('--I-x-h', metavar='', type=float, default=122.075, help='The highest longitude of input map. (default: 122.075)')
# parser.add_argument('--I-y-l', metavar='', type=float, default=24.4375, help='The lowest latitude of input map. (default: 24.4375)')
# parser.add_argument('--I-y-h', metavar='', type=float, default=25.55, help='The highest latitude of input map. (default: 25.55)')

# parser.add_argument('--F-x-l', metavar='', type=float, default=121.3375, help='The lowest longitude of target map. (default: 121.3375)')
# parser.add_argument('--F-x-h', metavar='', type=float, default=121.7, help='The highest longitude of target map. (default: 121.7)')
# parser.add_argument('--F-y-l', metavar='', type=float, default=24.8125, help='The lowest latitude of target map. (default: 24.8125)')
# parser.add_argument('--F-y-h', metavar='', type=float, default=25.175, help='The highest latitude of target map. (default: 25.175)')

# tw forecast size
parser.add_argument('--I-x-l', metavar='', type=float, default=118.8, help='The lowest longitude of input map. (default: 118.3)')
parser.add_argument('--I-x-h', metavar='', type=float, default=122.5375, help='The highest longitude of input map. (default: 123.2875)')
parser.add_argument('--I-y-l', metavar='', type=float, default=21.75, help='The lowest latitude of input map. (default: 21)')
parser.add_argument('--I-y-h', metavar='', type=float, default=25.4875, help='The highest latitude of input map. (default: 25.9875)')

parser.add_argument('--F-x-l', metavar='', type=float, default=118.8, help='The lowest longitude of target map. (default: 118.3)')
parser.add_argument('--F-x-h', metavar='', type=float, default=122.5375, help='The highest longitude of target map. (default: 123.2875)')
parser.add_argument('--F-y-l', metavar='', type=float, default=21.75, help='The lowest latitude of target map. (default: 21)')
parser.add_argument('--F-y-h', metavar='', type=float, default=25.4875, help='The highest latitude of target map. (default: 25.9875)')

parser.add_argument('--O-x-l', metavar='', type=float, default=118, help='The lowest longitude of original map. (default: 118)')
parser.add_argument('--O-x-h', metavar='', type=float, default=123.5, help='The highest longitude of original map. (default: 123.5)')
parser.add_argument('--O-y-l', metavar='', type=float, default=20, help='The lowest latitude of original map. (default: 20)')
parser.add_argument('--O-y-h', metavar='', type=float, default=27, help='The highest latitude of original map. (default: 27)')

parser.add_argument('--weather-list', metavar='', action='append', default=[],
                    help='Weather list. (default: [])')

args = parser.parse_args()

if args.able_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

if args.dtype == 'float16':
    args.value_dtype = torch.float16
    args.optimizer = optim.SGD
elif args.dtype == 'float32':
    args.value_dtype = torch.float32
    args.optimizer = optim.SGD

args.res_degree = 0.0125
args.I_x = [args.I_x_l, args.I_x_h]
args.I_y = [args.I_y_l, args.I_y_h]
args.F_x = [args.F_x_l, args.F_x_h]
args.F_y = [args.F_y_l, args.F_y_h]
args.O_x = [args.O_x_l, args.O_x_h]
args.O_y = [args.O_y_l, args.O_y_h]

args.I_shape = (round((args.I_x_h-args.I_x_l)/args.res_degree)+1, round((args.I_y_h-args.I_y_l)/args.res_degree)+1)
args.F_shape = (round((args.F_x_h-args.F_x_l)/args.res_degree)+1, round((args.F_y_h-args.F_y_l)/args.res_degree)+1)
args.O_shape = (round((args.O_x_h-args.O_x_l)/args.res_degree)+1, round((args.O_y_h-args.O_y_l)/args.res_degree)+1)

# overall info for normalization
rad_overall = pd.read_csv(os.path.join(args.radar_folder, 'overall.csv'), index_col='Measures').loc['max':'min',:]
meteo_overall = pd.read_csv(os.path.join(args.weather_folder, 'overall.csv'), index_col='Measures')
args.max_values = pd.concat([rad_overall, meteo_overall], axis=1, sort=False).T['max']
args.min_values = pd.concat([rad_overall, meteo_overall], axis=1, sort=False).T['min']

# # args.I_x_iloc = [int((args.I_x[0]-args.O_x[0])/args.res_degree), int((args.I_x[1]-args.O_x[0])/args.res_degree + 1)]
# # args.I_y_iloc = [int((args.I_y[0]-args.O_y[0])/args.res_degree), int((args.I_y[1]-args.O_y[0])/args.res_degree + 1)]
# # args.F_x_iloc = [int((args.F_x[0]-args.O_x[0])/args.res_degree), int((args.F_x[1]-args.O_x[0])/args.res_degree + 1)]
# # args.F_y_iloc = [int((args.F_y[0]-args.O_y[0])/args.res_degree), int((args.F_y[1]-args.O_y[0])/args.res_degree + 1)]

args.compression = 'bz2'
args.figure_dpi = 120

args.RAD_level = [-5, 0, 10, 20, 30, 40, 50, 60, 70]
args.QPE_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]
args.QPF_level = [-5, 0, 10, 20, 35, 50, 80, 120, 160, 200]

args.RAD_cmap = ['#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000','#600000']
args.QPE_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030']
args.QPF_cmap = ['#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000050','#000030']
args.input_channels = 1 + args.input_with_QPE*1 + len(args.weather_list) + args.input_with_grid*2
# # args.xaxis_list = np.around(np.linspace(args.I_x[0], args.I_x[1], args.I_shape[0]), decimals=4)
# # args.yaxis_list = np.around(np.linspace(args.I_y[1], args.I_y[0], args.I_shape[1]), decimals=4)


if __name__ == '__main__':
    print(args.batch_norm)

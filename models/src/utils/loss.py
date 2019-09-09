import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def R2DBZ(x):
    return 10*np.log10(x**(8/5)*200)

def mse(x,y):
    return torch.sum((x-y)**2)

def mae(x,y):
    return torch.sum(torch.abs(x-y))

class Loss():
    def __init__(self, args):
        super().__init__()
        self.weights = np.array([1., 2., 5., 10., 20.])
        self.value_list = np.array([0., 2., 5., 10., 40., 200.])
        max_values = args.max_values['RAD']
        self.value_list[1:] = R2DBZ(self.value_list[1:])
        self.name = args.loss_function
        # breakpoint()

        if args.normalize_target:
            self.value_list = self.value_list / max_values
        
        if args.loss_function.upper() == 'BMSE':
            self.loss = F.mse_loss
        if args.loss_function.upper() == 'BMAE':
            self.loss = F.l1_loss
        if args.loss_function.upper() == 'MSE':
            self.loss = F.mse_loss
            self.weights = np.ones_like(self.weights)
        if args.loss_function.upper() == 'MAE':
            self.loss = F.l1_loss
            self.weights = np.ones_like(self.weights)

    def __call__ (self, outputs, targets):
        loss = 0.
        for i in range(len(self.weights)):
            mask = (targets>=self.value_list[i]) & (targets<self.value_list[i+1])
            tmp = self.weights[i] * self.loss(outputs[mask], targets[mask], reduction='mean')
            # make sure the loss is not nan value
            if torch.isnan(tmp):
                continue
            else:
                loss += tmp 
        return loss

class LOSS_pytorch():
    def __init__(self, args):
        super().__init__()
        self.weights = np.array([1., 2., 5., 10., 30., 100.])
        self.value_list = np.array([0., 2., 5., 10., 30., 60., 200.])
        max_values = args.max_values['QPE']

        if args.target_RAD:
            max_values = args.max_values['RAD']
            self.value_list[1:] = R2DBZ(self.value_list[1:])

        if args.normalize_target:
            self.value_list = self.value_list / max_values
        
        if args.loss_function.upper() == 'BMSE':
            self.loss = mse
        if args.loss_function.upper() == 'BMAE':
            self.loss = mae
        if args.loss_function.upper() == 'MSE':
            self.loss = mse
            self.weights = np.ones_like(self.weights)
        if args.loss_function.upper() == 'MAE':
            self.loss = mae
            self.weights = np.ones_like(self.weights)

    def __call__ (self, outputs, targets):
        loss = 0.
        b = outputs.shape[0]
        for i in range(len(self.weights)):
            mask = (targets>=self.value_list[i]) & (targets<self.value_list[i+1])
            loss += self.weights[i] * self.loss(outputs[mask], targets[mask]) / b
        return loss

class Criterion():
    def __init__(self, prediction, truth):
        value_list = [0, 2, 5, 10, 30, 500]
        self.prediction = prediction
        self.truth = truth
        self.batch_size = prediction.shape[0]

    def csi(self, threshold):
        TP_truth = self.truth>=threshold
        TP_predict = self.prediction>=threshold
        TP = np.sum(TP_truth == TP_predict)

        FN_truth = self.truth>=threshold
        FN_predict = self.prediction<threshold
        FN = np.sum(FN_truth == FN_predict)

        FP_truth = self.truth<threshold
        FP_predict = self.prediction>=threshold
        FP = np.sum(FP_truth == FP_predict)
        
        return TP/(TP+FN+FP)/self.batch_size

    def hss(self, threshold):
        TP_truth = self.truth>=threshold
        TP_predict = self.prediction>=threshold
        TP = np.sum(TP_truth == TP_predict)

        FN_truth = self.truth>=threshold
        FN_predict = self.prediction<threshold
        FN = np.sum(FN_truth == FN_predict)

        FP_truth = self.truth<threshold
        FP_predict = self.prediction>=threshold
        FP = np.sum(FP_truth == FP_predict)

        TN_truth = self.truth<threshold
        TN_predict = self.prediction<threshold
        TN = np.sum(TN_truth == TN_predict)

        return ((TP*TN)-(FN*FP)) / ((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)) / self.batch_size

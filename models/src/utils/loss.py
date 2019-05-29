import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MSE(nn.Module):
    def __init__(self, max_values, min_values, balance=True, normalize_target=False):
        super(MSE, self).__init__()
        if balance:
            self.weights = [1, 2, 5, 10, 30]
        else:
            self.weights = [1]*5
            
        if normalize_target:
            self.value_list = [0, 2/max_values, 5/max_values, 10/max_values, 30/max_values, 1]
        else:
            self.value_list = [0, 2, 5, 10, 30, 500]
        
    def forward(self, outputs, targets):
        loss = 0.
        for i in range(len(self.value_list)-1):
            mask = (targets>=self.value_list[i]) & (targets<self.value_list[i+1])
            tmp = self.weights[i] * F.mse_loss(outputs[mask], targets[mask])
            if torch.isnan(tmp):
                continue
            else:
                loss += tmp
        return loss

class MAE(nn.Module):
    def __init__(self, max_values, min_values, balance=True, normalize_target=False):
        super(MAE, self).__init__()
        if balance:
            self.weights = [1, 2, 5, 10, 30]
        else:
            self.weights = [1]*5
        if normalize_target:
            self.value_list = [0, 2/max_values, 5/max_values, 10/max_values, 30/max_values, 1]
        else:
            self.value_list = [0, 2, 5, 10, 30, 500]
        
    def forward(self, outputs, targets):
        loss = 0.
        for i in range(len(self.value_list)-1):
            mask = (targets>=self.value_list[i]) & (targets<self.value_list[i+1])
            tmp = self.weights[i] * F.l1_loss(outputs[mask], targets[mask])
            if torch.isnan(tmp):
                continue
            else:
                loss += tmp
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

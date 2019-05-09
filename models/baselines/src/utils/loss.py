import torch
import torch.nn as nn
import torch.nn.functional as F

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
            mask = torch.cat([(targets>=self.value_list[i]).unsqueeze(2), (targets<self.value_list[i+1]).unsqueeze(2)], dim=2).all(dim=2)
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
            mask = torch.cat([(targets>=self.value_list[i]).unsqueeze(2), (targets<self.value_list[i+1]).unsqueeze(2)], dim=2).all(dim=2)
            tmp = self.weights[i] * F.l1_loss(outputs[mask], targets[mask])
            if torch.isnan(tmp):
                continue
            else:
                loss += tmp
        return loss
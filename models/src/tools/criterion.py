import torch

def TP(target, pred, threshold):
    return torch.sum((target>=threshold) & (pred>=threshold))

def TN(target, pred, threshold):
    return torch.sum((target>=threshold) & (pred<threshold))

def FP(target, pred, threshold):
    return torch.sum((target<threshold) & (pred>=threshold))

def FN(target, pred, threshold):
    return torch.sum((target<threshold) & (pred<threshold))

def CSI(target, pred, threshold):
    '''Critical Success Index'''
    TP = TP(target, pred, threshold)
    TN = TN(target, pred, threshold)
    FP = FP(target, pred, threshold)
    return TP/(TN+TP+FP)

def FAR(target, pred, threshold):
    '''False Positive Rate'''
    FP = FP(target, pred, threshold)
    TN = TN(target, pred, threshold)
    return FP/(TN+FP)

def HSS(target, pred, threshold):
    TP = TP(target, pred, threshold)
    TN = TN(target, pred, threshold)
    FP = FP(target, pred, threshold)
    FN = FN(target, pred, threshold)
    return (TP*TN-FP*FN)/(TP+FN)*(FP+TN)+(TP+FP)*(FN+FP)


import torch

def BMSE(outputs, labels):
    bmse = 0
    outputs_size = outputs.shape[0]
    value_list = [0,2,5,10,30,500]
    for i in range(len(value_list)-1):
        chosen = torch.stack([value_list[i] <= labels, labels < value_list[i+1]]).all(dim=0)
        if i == 0:
            bmse += torch.sum(1*(outputs[chosen] - labels[chosen])**2)
        else:
            bmse += torch.sum(value_list[i]*(outputs[chosen] - labels[chosen])**2)

    return bmse/outputs_size

def BMAE(outputs, labels):
    bmae = 0
    outputs_size = outputs.shape[0]
    value_list = [0,2,5,10,30,500]
    for i in range(len(value_list)-1):
        chosen = torch.stack([value_list[i] <= labels, labels < value_list[i+1]]).all(dim=0)
        if i == 0:
            bmae += torch.sum(1*torch.abs(outputs[chosen] - labels[chosen]))
        else:
            bmae += torch.sum(value_list[i]*torch.abs(outputs[chosen] - labels[chosen])**2)

    return bmse/outputs_size

def BMAE(outputs, labels):
    BMAE = 0
    outputs_size = outputs.shape[0]*outputs.shape[1]
    BMAE += torch.sum(1*torch.abs(outputs[2>labels]-labels[2>labels]))
    BMAE += torch.sum(2*torch.abs(outputs[5>labels]-labels[5>labels])) - torch.sum(2*torch.abs(outputs[2>labels]-labels[2>labels]))
    BMAE += torch.sum(5*torch.abs(outputs[10>labels]-labels[10>labels])) - torch.sum(5*torch.abs(outputs[5>labels]-labels[5>labels]))
    BMAE += torch.sum(10*torch.abs(outputs[30>labels]-labels[30>labels])) - torch.sum(10*torch.abs(outputs[10>labels]-labels[10>labels]))
    BMAE += torch.sum(30*torch.abs(outputs[labels>=30]-labels[labels>=30]))

    return BMAE/outputs_size

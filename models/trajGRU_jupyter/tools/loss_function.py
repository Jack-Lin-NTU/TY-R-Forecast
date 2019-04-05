import torch


# def BMAE(outputs, labels):
#     BMAE = 0
#     outputs_size = outputs.shape[0]*outputs.shape[1]
#     BMAE += torch.sum(1*torch.abs(outputs[2>labels]-labels[2>labels]))
#     BMAE += torch.sum(2*torch.abs(outputs[5>labels]-labels[5>labels])) - torch.sum(2*torch.abs(outputs[2>labels]-labels[2>labels]))
#     BMAE += torch.sum(5*torch.abs(outputs[10>labels]-labels[10>labels])) - torch.sum(5*torch.abs(outputs[5>labels]-labels[5>labels]))
#     BMAE += torch.sum(10*torch.abs(outputs[30>labels]-labels[30>labels])) - torch.sum(10*torch.abs(outputs[10>labels]-labels[10>labels]))
#     BMAE += torch.sum(30*torch.abs(outputs[labels>=30]-labels[labels>=30]))

#     return BMAE

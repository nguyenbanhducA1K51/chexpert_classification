import cv2
import numpy as  np
import torchvision.transforms as transform
from easydict import EasyDict as edict
import json,os
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
import torch.nn as nn
class balanceBCE(nn.Module):
    def __init__(self,beta,device):
        super(balanceBCE,self).__init__()
        self.beta=beta
        self.device=device
    def forward(self,output,target):
        output=torch.sigmoid(output)
    
        output=torch.clamp(output,min=1e-10, max=0.9999999)

        positive=torch.sum(target,dim=0)
        negative= target.size()[0]-positive
        positive_factor= (1-self.beta)/ (1-self.beta**positive+1e-10)

        negative_factor=(1-self.beta)/ (1-self.beta**negative+1e-10)

        positive_factor=  positive_factor.unsqueeze(0).to(self.device)
        negative_factor= negative_factor.unsqueeze(0).to(self.device)

        positive_factor=torch.repeat_interleave(positive_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)
        negative_factor=torch.repeat_interleave(negative_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)

        # print ("positive_factor (1-beta)/ (1-beta**positive+1e-10) \n{}".format(positive_factor))
        # print ("negative_factor (1-beta)/ (1-beta**negative+1e-10)\n{}".format(negative_factor))
        loss=-positive_factor*target*torch.log(output)-(1-target)* negative_factor*torch.log(1-output)
        return loss

#output=torch.rand(5,6)
# target=torch.randint(0,2,(5,6))
# B=0.5
# res=balanceCE(output,target,device="cpu",beta=B)


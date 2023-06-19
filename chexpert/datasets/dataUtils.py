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
def balanceCE(y_score,y_true,device,beta):
   
    y_score=torch.sigmoid(y_score)
    # print ("y_true \n{}".format(y_true))
    # print ("y_pred \n{}".format(y_score))
    y_score=torch.clamp(y_score,min=1e-10, max=0.9999999)

    positive=torch.sum(y_true,dim=0)
    negative= y_true.size()[0]-positive
    positive_factor= (1-beta)/ (1-beta**positive+1e-10)

    negative_factor=(1-beta)/ (1-beta**negative+1e-10)

    positive_factor=  positive_factor.unsqueeze(0).to(device)
    negative_factor= negative_factor.unsqueeze(0).to(device)

    positive_factor=torch.repeat_interleave(positive_factor, torch.tensor([y_true.size()[0]]).to(device), dim=0)
    negative_factor=torch.repeat_interleave(negative_factor, torch.tensor([y_true.size()[0]]).to(device), dim=0)

    # print ("positive_factor\n{}".format(positive_factor))
    # print ("negative_factor\n{}".format(negative_factor))
    loss=-positive_factor*y_true*torch.log(y_score)-(1-y_true)* negative_factor*torch.log(1-y_score)
    return loss.to(device)

y_score=torch.rand(5,6)
y_true=torch.randint(0,2,(5,6))
B=0.5
res=balanceCE(y_score,y_true,device="cpu",beta=B)


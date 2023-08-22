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
from typing import Union, Tuple, List,Literal
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
class balanceBCE(nn.Module):
    def __init__(self,beta,device):
        super(balanceBCE,self).__init__()
        self.beta=beta
        self.device=device
    def forward(self,output,target):
        output=torch.sigmoid(output)
        ep=1e-10
        output=torch.clamp(output,min=ep, max=1-ep)
        positive=torch.sum(target,dim=0)
        negative= target.size()[0]-positive
        positive_factor= (1-self.beta)/ (1-self.beta**positive+1e-10)
        negative_factor=(1-self.beta)/ (1-self.beta**negative+1e-10)
        positive_factor=  positive_factor.unsqueeze(0).to(self.device)
        negative_factor= negative_factor.unsqueeze(0).to(self.device)
        positive_factor=torch.repeat_interleave(positive_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)
        negative_factor=torch.repeat_interleave(negative_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)
        loss=-positive_factor*target*torch.log(output)-(1-target)* negative_factor*torch.log(1-output)
        return loss

def visualize_images_mulitple_times(dataloader,img_list_idx:List[int], times:int,save_path):
        plt.figure(figsize=(30, 30))
        fig,ax =plt.subolots(len(img_list_idx),times, fig_size=(30,30))
        

   
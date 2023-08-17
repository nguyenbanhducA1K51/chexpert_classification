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
      
    return 

    # def transform(image, cfg):
    #     assert image.ndim == 2, "image must be gray image"
    #     if cfg.use_equalizeHist:
    #         image = cv2.equalizeHist(image)

    #     if cfg.gaussian_blur > 0:
    #         image = cv2.GaussianBlur(
    #             image,
    #             (cfg.gaussian_blur, cfg.gaussian_blur), 0)

    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #     image = fix_ratio(image, cfg)
    #     # augmentation for train or co_train

    #     # normalization
    #     image = image.astype(np.float32) - cfg.pixel_mean
    #     # vgg and resnet do not use pixel_std, densenet and inception use.
    #     if cfg.pixel_std:
    #         image /= cfg.pixel_std
    #     # normal image tensor :  H x W x C
    #     # torch image tensor :   C X H X W
    #     image = image.transpose((2, 0, 1))

    #     return image

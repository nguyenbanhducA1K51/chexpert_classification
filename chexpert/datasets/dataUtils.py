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

def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.image.image_fix_length
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.image.image_fix_length
        h_ = round(w_ / ratio)
    # print(" h {} w{}".format(h_,w_ ))
    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image=np.pad(image, ((0, cfg.image.image_fix_length - image.shape[0]),
                               (0, cfg.image.image_fix_length - image.shape[1]), (0, 0)),
                       mode='constant',
                       constant_values=cfg.image.pixel_mean)
    # print(image.shape)
    return image

def defaultTransform(image, cfg):
    # Transformimage in form of numpy array
    assert image.ndim == 2, "image must be gray image"

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, cfg)
    # normalization
    image = image.astype(np.float32) - cfg.image.pixel_mean
    if cfg.image.pixel_std:
        image /= cfg.image.pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image
def trainTransform(image):
    aug=transform.Compose([
        # transform.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
        #                  scale=(0.95, 1.05), fillcolor=128)
        transform.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05))
        # gaussian noise,.etc
        # 
    ])
    image=aug(image)
    return image


# def balanceCE(y_score,y_pred):
#     return
def balanceCE(y_score,y_true,device):
    y_score=torch.sigmoid(y_score)
    y_score=torch.clamp(y_score,min=1e-10, max=0.9999999)
    oneMinusB=torch.sum(y_true,dim=0)/y_true.size()[0]
    
    oneMinusB= oneMinusB.unsqueeze(0).to(device)

    oneMinusB=torch.repeat_interleave(oneMinusB, torch.tensor([y_true.size()[0]]).to(device), dim=0)
    
    B= (1- oneMinusB)
    oneMinusB=torch.clamp(oneMinusB,min=1e-10, max=0.9999999)
    B=torch.clamp(B,min=1e-10, max=0.9999999)
    loss=-B*y_true*torch.log(y_score)-(1-y_true)*oneMinusB*torch.log(1-y_score)
    return loss.to(device)


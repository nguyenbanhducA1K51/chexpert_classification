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

def visualize_images_mulitple_times(dataloader,img_list_idx:List[int], times:int,save_path):
        plt.figure(figsize=(30, 30))
        fig,ax =plt.subolots(len(img_list_idx),times, fig_size=(30,30))
        

   
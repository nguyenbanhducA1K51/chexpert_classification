from math import exp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
sys.path.append("../datasets")
sys.path.append("../model")
import data
import utils
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import random_split
    
class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.softmax=nn.Softmax(dim=2)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[6] = utils.MultiDimLinear( in_features=4096, out_shape=(num_classes, 3))
        for name, param in self.vgg.named_parameters():
                if name!="classifier.6.weight" and name!="classifier.6.bias":
                    param.requires_grad = False
         
    def forward(self, x):
        x= self.vgg(x)
        x=self.softmax(x)
        return x
   






import glob
import os
import pandas as pd 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from math import exp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import data
from torch.nn import functional as F
from torch.utils.data import random_split
     
class ChestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform, numPatient=30):
        """
        Arguments:
            
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            numPatient (int) : number of Patient when working on a subset of dataset
        """
        
        self.root_dir = root_dir

        self.num_patient= numPatient

        self.transform = transform
        #  start with 1 because it has the file .DS_Store in the begin
        subdir= sorted(os.listdir(root_dir))[1:numPatient+1]
        self.img_path=[]
        for dir in subdir:
            self.img_path+= glob.glob("{}/{}/*/*.jpg".format(root_dir,dir))  
        # add 1 because the first row is header
        self.labels = pd.read_csv(csv_file, nrows=len(self.img_path)+1)
        # exclude the first row of header
        self.labels=np.array(self.labels.iloc[1:,5:])
        self.numclass=self.labels.shape[1]
       
        is_nan=np.isnan(self.labels)
        # convert NaN values to 2
        self.labels[is_nan]=0
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
       
        label=self.labels[idx]
        im = Image.open(self.img_path[idx]) 
        im = self.transform(im)
        return im,label
    def getNumClass(self):
        return self.numclass

def loadData(train_csv_path,train_image_path, test_csv_path,
test_image_path,numPatient,transform,validation_split = .4,batch_size = 1
):
    trainset=ChestDataset(train_csv_path,train_image_path,transform=transform["train"],numPatient=numPatient)
    valtestset=ChestDataset(test_csv_path,test_image_path,transform=transform["val"],numPatient=numPatient)
    test_data, val_data = random_split(valtestset, [1-validation_split, validation_split])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=1)                     

    test_loader=torch.utils.data.DataLoader(test_data, batch_size=1)  
    numclass= trainset.getNumClass()
    return numclass,train_loader,validation_loader,test_loader


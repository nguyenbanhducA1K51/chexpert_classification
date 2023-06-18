import glob
import os
import pandas as pd 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random 
import sys
sys.path.append("../datasets")
import dataUtils
import torch
import numpy as np
import cv2
import os
from torch.nn import functional as F
from torch.utils.data import random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

class ChestDataset(Dataset):

    def __init__(self,cfg, csv_file, root_dir, transform=None, numPatient=30,mode="train"):
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
        self.cfg=cfg
        self.mode=mode
        self.transform=transform
        #  start with 1 because it has the file .DS_Store in the begin
        subdir= sorted(os.listdir(root_dir))[1:numPatient+1]
        self.img_path=[]
        for dir in subdir:
            self.img_path+= glob.glob("{}/{}/*/*.jpg".format(root_dir,dir))  
  
        self.labels = pd.read_csv(csv_file, nrows=len(self.img_path)+1)
        # print (  self.labels.columns.tolist())
        self.labels=np.array(self.labels.iloc[0:,5:])
        self.numclass=self.labels.shape[1]
       
        is_nan=np.isnan(self.labels)
        # convert NaN values to 0
        self.labels[is_nan]=0
        self.labels[self.labels==-1]=1
        self.csvPath=csv_file
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
       
        label=self.labels[idx]
        # 0 is a must so that cv read with number of channel is 1, or gray mode
        # image=cv2.imread(self.img_path[idx],0)
        image = Image.open(self.img_path[idx])
        if self.transform is not None:
            # image = self.transform(image=image)["image"]
            # image=self.transform(image=image)
            image=self.transform(image)    
        # return image['image'],label
        return image,label

    def getNumClass(self):
        return self.numclass

    def getsample(self,idx):
        print (self.img_path[idx])
        image=cv2.imread(self.img_path[idx],0)
        image = Image.fromarray(image)
        image.show()
        print ("label {}".format(self.labels[idx]) )
      
    def getRandomSample(self,num):
      
        for i in range (num):
            length=len(self.img_path)
            idx=random.randint(0,length) 
            image=cv2.imread(self.img_path[idx],0)
            image = Image.fromarray(image)
            image.show()
       
def loadData(cfg,train_csv_path,train_image_path, test_csv_path,
test_image_path,numPatient,validation_split = .4,batch_size = 1
):
#     train_transform = A.Compose([
#     A.RandomResizedCrop(320,320),
#     lambda x: torch.from_numpy( x.copy().float().div(255).unsqueeze(0)),   # tensor in [0,1]
#     A.Normalize(mean=[0.5330], std=[0.0349]),   
#      A.augmentations.geometric.transforms.Affine(
#     translate_percent=(-0.2, 0.2),
#     rotate=(-15, 15),
#     shear=(-10, 10),
#     p=0.5
# )  ,                             
#     lambda x: x.expand(3,-1,-1)    ])   
    
#     val_transform = A.Compose([
#       A.Resize(320,320),
#     lambda x: torch.from_numpy(x.copy().float().div(255).unsqueeze(0) ),   # tensor in [0,1]
#     A.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
#     lambda x: x.expand(3,-1,-1)])  


    train_transform = T.Compose([
        T.RandomResizedCrop(320),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
        T.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
        lambda x: x.expand(3,-1,-1)])  
    val_transform = T.Compose([      
        T.Resize(320),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
        T.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
        lambda x: x.expand(3,-1,-1)]) 

    trainset=ChestDataset(cfg=cfg,csv_file=train_csv_path,root_dir=train_image_path,numPatient=numPatient["train"],mode="train", transform=train_transform)

    valtestset=ChestDataset(cfg=cfg,csv_file=test_csv_path,root_dir=test_image_path,numPatient=numPatient["val"],mode="val",transform=val_transform)
    test_data, val_data = random_split(valtestset, [1-validation_split, validation_split])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=1)                     

    test_loader=torch.utils.data.DataLoader(test_data, batch_size=1)  
    numclass= trainset.getNumClass()

    return numclass,train_loader,validation_loader,test_loader



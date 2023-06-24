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
from easydict import EasyDict as edict
import json,os

class ChestDataset(Dataset):
   
    def __init__(self,disease,root,csv_file,transform=None, mini_data=None):
        self.disease=disease
        self.root=root
        self.transform=transform
        self.df=pd.read_csv(csv_file)
        
        self.attr_idxs = [self.df.columns.tolist().index(a) for a in self.disease]
        self.labels=self.df.iloc[:,self. attr_idxs]
        self.labels.fillna(0, inplace=True)
        self.labels.replace(-1,1,inplace=True)
        if mini_data is not None:
            self.length_sample=min ( mini_data, len(self.df))
        else:
            self.length_sample=len(self.df)
        
    def __len__(self):
        return self.length_sample
    def __getitem__(self,idx):
        label=self.labels.iloc[idx]
        label=label.to_numpy()
        img_path=os.path.join(self.root,self.df.iloc[idx,0]) 
        image=cv2.imread(img_path,0)       
        if self.transform is not None:        
            image=self.transform(image=image)  
        return image["image"].reshape(3,320,320),label
    def getNumClass(self):
        return len(self.attr_idxs)

def loadData(cfg,mode="default"):
        disease=cfg.disease
        root=cfg.path.root
        train_csv_path=cfg.path.train_csv_path
        test_csv_path=cfg.path.test_csv_path
      
        if mode=="default":
                mini_data=cfg.mini_data
                batch_size = cfg.train.batch_size
                img_size=320
        elif mode=="progressive":
            mini_data=cfg.progressive_mini_data
            batch_size = cfg.progressive_train.batch_size
            img_size=cfg.image.progressive_image_size
        def norm (image,*arg,**karg):
            image= np.array(image,copy=True).astype(np.float32)/255

            return image
        def expand (image,*arg,**karg):
            image=np.expand_dims(image,axis=0)
            return np.repeat(image,3,axis=0)
        train_transform = A.Compose([
                            
                    
                     A.Lambda(name="normalize rangge (0,1)",image=norm),
#                     A.Resize(height=360,width=360),
                A.ShiftScaleRotate( scale_limit =((-0.2, 0.2)) ),
                     A.RandomSizedCrop(min_max_height=(300,310),height=320,width=320,w2h_ratio=1.0),
                       A.Normalize(mean=[0.5330], std=[0.0349],max_pixel_value=1.),

                      A.Lambda(name="expand dim", image=expand),
                     
                            ])
        val_transform=A.Compose([
                            A.Lambda(name="normalize rangge (0,1)",image=norm),  
                            A.Resize(height=320,width=320),
        
                           A.Normalize(mean=[0.5330], std=[0.0349],max_pixel_value=1.),
                            A.Lambda( name="expand dim",image=expand),
                                         
                            ])
        # print (train_transform)

        if mini_data is not None:
            trainset=ChestDataset(disease=disease,root=root,csv_file=train_csv_path,mini_data=mini_data["train"], transform=train_transform)

            testset=ChestDataset(disease=disease,root=root,csv_file=test_csv_path,mini_data=mini_data["val"],transform=val_transform)
        else:
            trainset=ChestDataset(disease=disease,root=root,csv_file=train_csv_path, transform=train_transform)

            testset=ChestDataset(disease=disease,root=root,csv_file=test_csv_path,transform=val_transform)
        # print (trainset.__getitem__(2)[0].shape)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
        # validation_loader = torch.utils.data.DataLoader(val_data, batch_size=1)                     

        test_loader=torch.utils.data.DataLoader(testset, batch_size=1)  
        numclass= trainset.getNumClass()

        return numclass,train_loader,test_loader

cfg_path="../config/config.json" 
with open(cfg_path) as f:
    cfg = edict(json.load(f))
numclass,train_loader,test_loader=loadData(cfg=cfg)
# path= "/root/CheXpert-v1.0-small/train/patient00002/study2/view1_frontal.jpg"

# print ("shape image {}".format(train_loader.dataset[0][0].shape) )

# print ("shape image {}".format(train_loader.dataset[0][1].shape) )

x,y= next(iter(train_loader))
print ("x ,y {} {}".format(x.size(),y.shape))





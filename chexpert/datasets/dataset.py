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
from common import csv_index

class ChestDataset(Dataset):  
    def __init__(self,cfg,csv_file,transform=None, mini_data=None,mode="train"):
        self.root=cfg.path.root      
        self.class_idx=cfg.class_idx
        self.col_names=[]
        for idx in self.class_idx:
            self.col_names.append(csv_index[str(idx)])
        self.transform=transform
        self.df=pd.read_csv(csv_file)   
        self.df["orinIndex"]=self.df.index    
        self.attr_idxs = [self.df.columns.tolist().index(a) for a in self.disease]   
        if mini_data is not None:
            self.length_sample=min ( mini_data, len(self.df))
        else:
            self.length_sample=len(self.df)
        self.idx=np.random.choice(np.arange(0,len(self.df)), size=self.length_sample)
        self.df=self.df.iloc[self.idx].reset_index(drop=True)
        self.labels=self.df.loc[self.col_names]
        print (self.labels.head())
        self.labels.fillna(0, inplace=True)
        self.labels.replace(-1,1,inplace=True)
        
    def __len__(self):
        return self.length_sample
    def __getitem__(self,idx):
        label=self.labels.iloc[idx]
        label=label.to_numpy()
        img_path=os.path.join(self.root,self.df.iloc[idx,0]) 
        image=cv2.imread(img_path,0)       
        if self.transform is not None:        
            image=self.transform(image=image)  
        return image["image"],label
   
def loadData(cfg,mode="default"):
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
       
        def expand (image,*arg,**karg):
            image=np.expand_dims(image,axis=0)
            return np.repeat(image,3,axis=0)
        
        factor=0.05
        ceil=int (img_size*(1+factor) )
        floor=int (img_size*(1-factor) )
        train_transform = A.Compose([
                A.Resize(height=ceil,width=ceil) ,                     
                A.ShiftScaleRotate( scale_limit =((-0.2, 0.2)) ),
                A.RandomSizedCrop(min_max_height=(floor,ceil ),height=img_size,width=img_size),
                A.Normalize(mean=[0.5330], std=[0.0349]),
                A.Lambda( image=expand),                
                            ])
        val_transform=A.Compose([  
                A.Resize(height=img_size,width=img_size),
                A.Normalize(mean=[0.5330], std=[0.0349]),
                A.Lambda( image=expand),
                                 ])  
        traindata=ChestDataset(cfg=cfg,csv_file=train_csv_path,mini_data=mini_data["train"] , transform=train_transform,mode="train")
        testdata=ChestDataset(cfg=cfg,csv_file=test_csv_path,mini_data=mini_data["val"] ,transform=val_transform,mode="val")
    
        return traindata,testdata
def tta_transform():
    def expand (image,*arg,**karg):
            image=np.expand_dims(image,axis=0)
            return np.repeat(image,3,axis=0)
    transform = A.Compose([             
                A.ShiftScaleRotate( scale_limit =((-0.2, 0.2)) ),
                A.RandomSizedCrop(min_max_height=(300,310),height=320,width=320,w2h_ratio=1.0),
                A.Normalize(mean=[0.5330], std=[0.0349]),
                A.Lambda(name="expand dim", image=expand),                  
                            ])
    return transform
cfg_path="/root/repo/Chexpert/chexpert/config/config.json" 
with open(cfg_path) as f:
    cfg = edict(json.load(f))
traindata,testdata=loadData(cfg=cfg)
# path= "/root/CheXpert-v1.0-small/train/patient00002/study2/view1_frontal.jpg"

# print ("shape image {}".format(train_loader.dataset[0][0].shape) )

# print ("shape image {}".format(train_loader.dataset[0][1].shape) )

# x,y= next(iter(train_loader))
# print ("x ,y {} {}".format(x.size(),y.shape))




print (csv_index)

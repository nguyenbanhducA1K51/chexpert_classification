import os
import pandas as pd 
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append("/root/repo/chexpert_classification/chexpert/")
from data import dataUtils
import torch
import numpy as np
import cv2
import os
import albumentations as A
import json,os
from data.common import csv_index,load_transform
from typing import Union, Tuple, List,Literal
import cv2
class ChestDataset(Dataset):  
    def __init__(self,cfg, mode:Literal["train","test"]="train",train_mode:Literal["default","progressive"]="default"):
        if mode=="train":
            csv_file=cfg.path.train_csv_path
            mini_data=cfg.mini_data.train
        elif mode=="test":
            csv_file=cfg.path.test_csv_path
            mini_data=cfg.mini_data.test
        else:
            raise RuntimeError("Invalid mode")
        
        self.transform=load_transform(cfg,mode=mode,train_mode=train_mode)
        self.root=cfg.path.root      
        self.class_idx=cfg.class_idx
       
        self.df=pd.read_csv(csv_file)   
        self.columns= list(self.df.columns)
        self.class_=[]
        # for i in range(len(self.class_idx)):
        #      print (f"classifying class  {self.columns[self.class_idx[i]]}")
        #      self.class_.append(self.columns[self.class_idx[i]])

        self.df["orinIndex"]=self.df.index          
        if mini_data ==-1:
            self.length_sample=len(self.df)         
        else:
             self.length_sample=min ( mini_data, len(self.df))
        self.idx=np.random.choice(np.arange(0,len(self.df)), size=self.length_sample)
      
        self.df=self.df.iloc[self.idx].reset_index(drop=True)
        self.df.fillna(0, inplace=True)
        self.df.replace(-1,1,inplace=True)
        # print (self.df.info())

        # for column in self.class_:
        #     unique_values = self.df[column].unique()
        #     print(f"Unique values in '{column}': {unique_values}")
        #     positive=np.sum(np.array(self.df[column]))
        #     print (f"{column} total positive {positive}, ration {positive/len(self.df)}")
        
    def __len__(self):
        return self.length_sample
    def __getitem__(self,idx):
        label=self.df.iloc[idx,self.class_idx]
        label=label.to_numpy().astype(int)
        img_path=os.path.join(self.root,self.df.iloc[idx,0]) 
        image=cv2.imread(img_path,0)       
        if self.transform is not None: 
        
            image=self.transform(image=image)  
        return  image["image"],label
    
    def calculate_mean_std(self,samples:int):
        means=[]
        variations=[]
        if samples==-1:
            iteration=self.length_sample
        else:
            iteration=min(self.length_sample,samples)
        for i in range(iteration):
            print (f"find mean, process {i}/{iteration}")
            img_path=os.path.join(self.root,self.df.iloc[i,0]) 
            image=cv2.imread(img_path,0)    
            means.append(np.mean(image))
        mean=np.mean(means)
        for i in range(iteration):
            print (f"find std, process {i}/{iteration}")
            img_path=os.path.join(self.root,self.df.iloc[i,0]) 
            image=cv2.imread(img_path,0)    
            image=(image-mean)**2
            variations.append(np.sum(image))
        sample=cv2.imread(os.path.join(self.root,self.df.iloc[0,0]) ,0)  
        shape=sample.shape
        var=np.sum(variations)/ (iteration*np.prod(shape))
        std=np.sqrt(var)
        return mean,std

         
           
# def loadData(cfg,mode="default"):
#     if mode=="default":
#         mini_data=cfg.mini_data
#         batch_size = cfg.train.batch_size
#         img_size=320
#     elif mode=="progressive":
#         mini_data=cfg.progressive_mini_data
#         batch_size = cfg.progressive_train.batch_size
#         img_size=cfg.image.progressive_image_size
    
#     train_data=ChestDataset(cfg=cfg,mini_data=mini_data["train"] , mode="train")
#     test_data=ChestDataset(cfg=cfg,mode="test")
#     train_loader=torch.utils.data.DataLoader (train_data,batch_size=batch_size,shuffle=True,num_workers=16)
#     test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=False,num_workers=16)
#     return train_loader,test_loader

# def tta_loader(cfg):
#     def expand (image,*arg,**karg):
#             image=np.expand_dims(image,axis=0)
#             return np.repeat(image,3,axis=0)   
#     img_size=320    
#     factor=0.05
#     ceil=int (img_size*(1+factor) )
#     floor=int (img_size*(1-factor) )
#     test_transform = A.Compose([
#             A.Resize(height=ceil,width=ceil) ,                     
#             A.ShiftScaleRotate( scale_limit =((-0.2, 0.2)) ),
#             A.RandomSizedCrop(min_max_height=(floor,ceil ),height=img_size,width=img_size),
#             A.Normalize(mean=[128.21/255], std=[73.22/255]),
#             A.Lambda( image=expand),                
#                         ])
#     test_data=ChestDataset(cfg=cfg,mode="test")
#     return torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False, num_workers=16)   



if __name__=="__main__":
    from easydict import EasyDict as edict
    import json 
    cfg_path="/root/repo/chexpert_classification/chexpert/config/config.json"
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    dataset=ChestDataset(cfg=cfg,mode="train")
    for i in range (5):
        img,label=dataset.__getitem__(i)
        print ("type",type(img),type(label),label[i])
        print (label,label.dtype)


    # for i in range (10):
    #     img,label=dataset.__getitem__(i)
    #     print ("shape",img.shape,label, np.min(img),np.max(img),np.mean(img))

    # img1="/root/data/CheXpert-v1.0-small/train/patient10684/study10/view1_frontal.jpg"
    # assert os.path.exists(img1), "not exist"
    # x=cv2.imread(img1,0)
    # print (np.max(x),np.min(x),np.mean(x))
    # print (dataset.calculate_mean_std(-1))
    # train_loader,val_loader=loadData(cfg)

    

     






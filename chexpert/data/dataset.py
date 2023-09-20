import os
import pandas as pd 
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import cv2
import os
import albumentations as A
import json,os
from data.common import load_transform
from typing import Union, Tuple, List,Literal
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

class ChestDataset(Dataset):  
    def __init__(self,cfg,fold=1,mode:Literal["train","val","test"]="train",train_mode:Literal["default","progressive"]="default", view: Literal["frontal","lateral","both"]="frontal"):
        
        if mode!="test":

            csv_file=cfg["path"]["process_train"]
            mini_data=cfg["mini_data"]["train"]
        else:
            csv_file=cfg["path"]["process_test"]
            mini_data=cfg["mini_data"]["test"]
             
        self.transform=load_transform(cfg,mode=mode,train_mode=train_mode)
        self.root=cfg["path"]["data_path"]      
        self.class_idx= cfg["train_params"]["class_idx"].split(",")
        self.class_idx=[int(x) for x in self.class_idx]
            
        self.df=pd.read_csv(csv_file)  
        
        self.view=view

        if mode=="train":
            self.df['fold'] =  self.df['fold'].astype(int)
            self.df=self.df[self.df["fold"]!=int(fold)]
        elif mode=="val":
            self.df=self.df[self.df["fold"]==int(fold)]

        self.columns= list(self.df.columns)
        self.class_=[]
      
        self.df["orinIndex"]=self.df.index       

        if mini_data ==-1:
            self.length_sample=len(self.df)         
        else:
             self.length_sample=min ( mini_data, len(self.df))


        self.idx=np.random.choice(np.arange(0,len(self.df)), size=self.length_sample)

        self.df=self.df.iloc[self.idx].copy().reset_index(drop=True)
        
        if view=="frontal":
            self.df = self.df.loc[ self.df["is_frontal"] == 1].copy().reset_index(drop=True)
        elif view=="lateral":
            self.df = self.df.loc[self.df["is_frontal"] == 0].copy().reset_index(drop=True)
        # pd.set_option('display.max_columns', None)
        
        self.df.fillna(0, inplace=True)
        self.df.replace(-1,1,inplace=True)
      
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        label=self.df.iloc[idx,self.class_idx]
        label=label.to_numpy().astype(int)
        is_frontal= self.df.loc[idx,"is_frontal"]
        img_path=os.path.join(self.root,self.df.iloc[idx,0]) 
        image=cv2.imread(img_path,0)       
        if self.transform is not None:       
            image=self.transform(image=image)  

        return  image["image"],label,is_frontal,idx
    
      
    
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
              
def random_visualize(train_dataset,test_dataset):
    n_samples=10
  
    train_samples={}
    test_samples={}
    current_file_path = os.path.abspath(__file__)

    parent_directory = os.path.dirname(current_file_path)


    save_folder="/root/repo/chexpert_classification/chexpert/output/learning_analysis"
    for i in range (n_samples):

        train_idx=random.choice(range(len(train_dataset)))
        test_idx=random.choice(range(len(test_dataset)))
        train_samples[str(train_idx)]=train_dataset.__getitem__(train_idx)
        test_samples[str(test_idx)]=test_dataset.__getitem__(test_idx) 
    fig,ax=plt.subplots(4,n_samples, figsize=(30,30))
    plt.tight_layout()

    for i,key in enumerate (train_samples.keys()):
        ax[0,i].imshow(  train_samples[key][0][0],cmap="gray")
        ax[0,i].set_title(f"row {key} of train dataset" )

        ax[2,i].imshow(train_dataset.get_original_item(int(key))[0],cmap="gray")   
        ax[2,i].set_title(f"row {key} of original train dataset" )    

    for i,key in enumerate (test_samples.keys()):
        ax[1,i].imshow( test_samples[key][0][0], cmap="gray")
        ax[1,i].set_title(f"row {key} of test dataset" )

        ax[3,i].imshow(test_dataset.get_original_item(int(key))[0],cmap="gray")   
        ax[3,i].set_title(f"row {key} of original test dataset" )    


    plt.show()
    plt.savefig(os.path.join(save_folder,"samples_visualize.png"))
    plt.clf()



if __name__=="__main__":
    from easydict import EasyDict as edict
    import yaml
    from torch.utils.data import DataLoader

    config_path= Path(__file__).resolve().parent.parent /"config/config.yaml"
    with open(config_path, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    traindataset=ChestDataset(cfg=cfg,mode="train")
    latdataset=ChestDataset(cfg=cfg,mode="train",view="lateral")

    loader=DataLoader(traindataset,batch_size=5)
    for i , sample in enumerate(loader):
        pass

    # sam=traindataset.__getitem__(0)[0]
    # # print (sam.shape)
    # testdataset=ChestDataset(cfg=cfg,mode="test")
    # random_visualize(traindataset,testdataset)
   


    

     






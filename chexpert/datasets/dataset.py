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
    disease= [

    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
     "Atelectasis",
     "Pneumothorax",
    "Pleural Effusion",
     "Pleural Other",
     "Fracture",
     "Support Devices" 
]
    def __init__(self,root,csv_file,transform=None, mini_data=None):
        self.root=root
        self.transform=transform
        self.df=pd.read_csv(csv_file)
        self.attr_idxs = [self.df.columns.tolist().index(a) for a in self.disease]
        
        if mini_data is not None:
            self.length_sample=mini_data
        else:
            self.length_sample=len(self.df)
    def __len__(self):
        return self.length_sample
    def __getitem__(self,idx):
        label=self.df.iloc[idx,self.attr_idxs].to_numpy()
        img_path=os.path.join(self.root,self.df.iloc[idx,0])
        image = Image.open(img_path)
        if self.transform is not None:
           
            image=self.transform(image)  
        return image,label
    def getNumClass(self):
        return len(self.attr_idxs)

def loadData(root,train_csv_path, test_csv_path,
mini_data=None,validation_split = .4,batch_size = 1
):

    train_transform = T.Compose([
        T.RandomResizedCrop(320),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
        T.Normalize(mean=[0.5330], std=[0.0349]),   
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        
        lambda x: x.expand(3,-1,-1)])  
    val_transform = T.Compose([      
        T.Resize(320),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
        T.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
        lambda x: x.expand(3,-1,-1)]) 
    if mini_data is not None:
        trainset=ChestDataset(root,csv_file=train_csv_path,mini_data=mini_data["train"], transform=train_transform)

        valtestset=ChestDataset(root,csv_file=test_csv_path,mini_data=mini_data["val"],transform=val_transform)
    else:
        trainset=ChestDataset(root,csv_file=train_csv_path, transform=train_transform)

        valtestset=ChestDataset(root,csv_file=test_csv_path,transform=val_transform)

    test_data, val_data = random_split(valtestset, [1-validation_split, validation_split])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=1)                     

    test_loader=torch.utils.data.DataLoader(test_data, batch_size=1)  
    numclass= trainset.getNumClass()

    return numclass,train_loader,validation_loader,test_loader

cfg_path="../config/config.json" 
with open(cfg_path) as f:
    cfg = edict(json.load(f))

num_class, train_loader,val_loader,test_loader=loadData(root=cfg.path.root,train_csv_path=cfg.path.train_csv_path,
 test_csv_path=cfg.path.test_csv_path, mini_data=cfg.mini_data, validation_split = cfg.validation_split,batch_size = cfg.train.batch_size
)

print (train_loader.dataset[0][0].shape)





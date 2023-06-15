import glob
import os
import pandas as pd 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.append("../datasets")
import dataUtils
import torch
import numpy as np
import cv2
import os
from torch.nn import functional as F
from torch.utils.data import random_split
    
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
        self.transform = transform
        self.mode=mode
        #  start with 1 because it has the file .DS_Store in the begin
        subdir= sorted(os.listdir(root_dir))[1:numPatient+1]
        self.img_path=[]
        for dir in subdir:
            self.img_path+= glob.glob("{}/{}/*/*.jpg".format(root_dir,dir))  
  
        self.labels = pd.read_csv(csv_file, nrows=len(self.img_path)+1)
        print (  self.labels.columns.tolist())
        self.labels=np.array(self.labels.iloc[0:,5:])
        self.numclass=self.labels.shape[1]
       
        is_nan=np.isnan(self.labels)
        # convert NaN values to 0
        self.labels[is_nan]=0
        self.labels[self.labels==-1]=0
        self.csvPath=csv_file
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
       
        label=self.labels[idx]
        # 0 is a must so that cv read with number of channel is 1, or gray mode
        image=cv2.imread(self.img_path[idx],0)
        #convert numpy array image to PIL image for transformation
        image = Image.fromarray(image)
        if self.mode=="train":
            image=dataUtils.trainTransform(image=image)

        image=np.array(image)
        image=dataUtils.defaultTransform(image,cfg=self.cfg)
        return image,label

    def getNumClass(self):
        return self.numclass

    def getsample(self,idx):
        print (self.img_path[idx])
        image=cv2.imread(self.img_path[idx],0)
        image = Image.fromarray(image)
        image.show()
        print ("label {}".format(self.labels[idx]) )
        # csv = pd.read_csv(self.csvPath, nrows=5)
        # # exclude the first row of header
        # print (np.array(csv.iloc[:,0:]) )
        # print (np.array(csv.iloc[:,5:]) )
        # print (np.array(csv.iloc[1:,5:]) )
        
def loadData(cfg,train_csv_path,train_image_path, test_csv_path,
test_image_path,numPatient,validation_split = .4,batch_size = 1
):
    trainset=ChestDataset(cfg=cfg,csv_file=train_csv_path,root_dir=train_image_path,numPatient=numPatient["train"],mode="train")
    # trainset.getsample(5)
    valtestset=ChestDataset(cfg=cfg,csv_file=test_csv_path,root_dir=test_image_path,numPatient=numPatient["val"],mode="val")
    test_data, val_data = random_split(valtestset, [1-validation_split, validation_split])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=1)                     

    test_loader=torch.utils.data.DataLoader(test_data, batch_size=1)  
    numclass= trainset.getNumClass()

    return numclass,train_loader,validation_loader,test_loader



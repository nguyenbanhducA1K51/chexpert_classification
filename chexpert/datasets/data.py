import glob
import os
import pandas as pd 
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
class ChestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform, numPatient=30):
        """
        Arguments:
            
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir

        self.num_patient= numPatient

        self.transform = transform
        #  start with 1 ? because it has the file .DS_Store in the begin
        subdir= sorted(os.listdir(root_dir))[1:numPatient+1]
        self.img_path=[]
        for dir in subdir:
            self.img_path+= glob.glob("{}/{}/*/*.jpg".format(root_dir,dir)) 
        
        # add 1 because the first row is header
        self.labels = pd.read_csv(csv_file, nrows=len(self.img_path)+1)
        # exclude the first row of header
        self.labels=np.array(self.labels.iloc[1:,5:])
        self.numclass=self.labels.shape[1]
        print (self.labels[0])
        is_nan=np.isnan(self.labels)
        # convert NaN values to 2
        self.labels[is_nan]=2
        print (self.labels[0])



    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
       
        label=self.labels[idx]
        im = Image.open(self.img_path[idx]) 
        im = self.transform(im)
        return im,label
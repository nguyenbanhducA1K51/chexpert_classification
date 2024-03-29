# index of class in csv file
import albumentations as A
import numpy as np
from typing import Union, Tuple, List,Literal
csv_index={
    "5": "No Finding",
    "6": "Enlarged Cardiomediastinum",
    "7": "Cardiomegaly",
    "8": "Lung Opacity",
    "9": "Lung Lesion",
    "10": "Edema",
    "11": "Consolidation",
    "12": "Pneumonia",
    "13": "Atelectasis",
    "14": "Pneumothorax",
    "15": "Pleural Effusion",
    "16": "Pleural Other",
    "17": "Fracture",
    "18": "Support Devices"
}

def load_transform(cfg,mode:Literal["train","test"]="train", train_mode:Literal["default","progressive"]="default") :     
    factor=0.05
    if train_mode=="default":           
        img_size=cfg ["image"]["image_fix_length"]
    elif train_mode=="progressive":
        img_size=cfg ["image"]["image_fix_length"]//2
    else :
        raise RuntimeError(f"invalid train mode{train_mode} ")
          
    def expand (image,*arg,**karg):
                image=np.expand_dims(image,axis=0)
                return np.repeat(image,3,axis=0)
    train_transform = A.Compose([                   
                    A.ShiftScaleRotate( scale_limit =(-0.2, 0.2) ,rotate_limit=(-10,10)),
                    A.RandomResizedCrop(height=img_size,width=img_size,scale=(0.9, 1.0),ratio=(0.75, 1.3333333333333333)),
                    A.Resize(height=img_size,width=img_size),
                    A.HorizontalFlip(),
                    A.Normalize(mean=[128.21/255], std=[73.22/255]),
                    A.Lambda( image=expand),                
                                ])

    test_transform=A.Compose([  
                    A.Resize(height=img_size,width=img_size),
                    A.Normalize(mean=[128.21/255], std=[73.22/255]),
                    A.Lambda( image=expand),
                                    ])  
    if mode=="train":
          return train_transform
    elif mode=="test" or mode=="val":
          return test_transform
    else:
          raise RuntimeError("invalid mode ..data/common.py")


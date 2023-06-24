
import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
import dataset
import chexpert
from torch.nn import functional as F
from easydict import EasyDict as edict
cfg_path="../config/config.json" 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
with open(cfg_path) as f:
    cfg = edict(json.load(f))

num_class, train_loader,test_loader=dataset.loadData(cfg, mode="default")

# get the shape of first example 
# train_loader.dataset[0] will return the tuple of image and label, image is numpy array

# print (train_loader.dataset[0][0].shape)

model=chexpert.chexpertNet(cfg=cfg,device=device,num_class=num_class)
if cfg.train_mode.name=="default":
    model.train_epochs(train_loader=train_loader,val_loader=test_loader)
elif cfg.train_mode.name=="progressive":
    assert cfg.load_ckp=="False" , "load check point must be false"
    _, progress_train_loader,progress_test_loader=dataset.loadData(cfg, mode="progressive")
    model.progressive_train_epochs(train_loader=train_loader,val_loader=test_loader, progress_train_loader=progress_train_loader,progress_test_loader=progress_test_loader)










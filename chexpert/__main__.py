
import sys
import torch
import json,os
from  model import chexpert
from  datasets import dataset
from torch.nn import functional as F
from easydict import EasyDict as edict
cfg_path=os.path.dirname(os.path.abspath(__name__))+"/config/config.json"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(cfg_path) as f:
    cfg = edict(json.load(f))
# num_class, train_loader,test_loader=dataset.loadData(cfg, mode="default")

# get the shape of first example 
# train_loader.dataset[0] will return the tuple of image and label, image is numpy array
# print (train_loader.dataset[0][0].shape)

model=chexpert.chexpertNet(cfg=cfg,device=device)
if cfg.train_mode.name=="default":
    model.train_epochs()
elif cfg.train_mode.name=="progressive":
    assert cfg.load_ckp=="False" , "load check point must be false"
    model.progressive_train_epochs()










import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
import data
import chexpert
from torch.nn import functional as F
from easydict import EasyDict as edict
cfg_path="../config/config.json" 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


with open(cfg_path) as f:
    cfg = edict(json.load(f))

num_class, train_loader,val_loader,test_loader=data.loadData(cfg=cfg,train_csv_path=cfg.path.train_csv_path,train_image_path=cfg.path.train_image_path,
 test_csv_path=cfg.path.test_csv_path,test_image_path=cfg.path.test_image_path,numPatient=cfg.numPatient, validation_split = cfg.validation_split,batch_size = cfg.train.batch_size
)
print (len(train_loader.dataset))
model=chexpert.chexpertNet(cfg=cfg,device=device)
model.train_epochs(train_data=train_loader,val_data=val_loader)
model.test(test_data=test_loader)


# torch.load("/Users/mac/vinBrain/cvDucnguyen/chexpert/model/output/best_model.pth")





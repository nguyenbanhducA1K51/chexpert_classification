
import sys
import torch
import json,os
# sys.path.append("../model")
# sys.path.append("../data")
from model import chexpert
from  data import dataset
from torch.nn import functional as F
from easydict import EasyDict as edict
cfg_path=os.path.dirname(os.path.abspath(__name__))+"/config/config.json"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(cfg_path) as f:
    cfg = edict(json.load(f))
net=chexpert.chexpertNet(cfg=cfg,device=device)
net.k_fold_train()
net.test()
# if cfg.train_mode.name=="default":
#     model.train_epochs()
# elif cfg.train_mode.name=="progressive":
#     assert cfg.load_ckp=="False" , "load check point must be false"
#     model.progressive_train_epochs()









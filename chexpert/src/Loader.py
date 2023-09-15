from pathlib import Path
# /root/repo/chexpert_classification/chexpert/config/config.yaml
import sys
sys.path.append(str( Path(__file__).resolve().parent))

from model import backbone
from src import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class Loader():
    def __init__(self,cfg,device="cpu"):

        self.device=device
        self.cfg=cfg
        self.model=self.loadModel()
        self.loss=self.loadLoss()
        self.optim,self.scheduler=self.loadOptimizer()
        if  cfg["train_mode"]=="progressive" :
            self.prog_optim,self.prog_scheduler=self.loadOptimizer() 
        else:
            self.prog_optim,self.prog_scheduler= None,None

    def loadModel(self):
        num_class= self.cfg["model"]["model_params"]["classes"]
        if self.cfg["model"]["name"]=="densenet121":    
            return backbone.DenseNetClassifier(num_classes=num_class,pretrain= True)
        elif self.cfg["model"]["name"]=="convnext_t":
            return backbone.ConvNextClassifier(num_classes=num_class,pretrain=True)
            
    def loadLoss(self):
        if "bce" in self.cfg["train_params"]["loss"]:
            return nn.BCEWithLogitsLoss(reduction='none')
        elif "balanceBCE" in  self.cfg["train_params"]["loss"]:
            return utils.balanceBCE(beta=self.cfg["train_params"]["loss"]["balanceBCE"]["beta"], device=self.device)
        else:
            raise Exception (" not support that criterion")
                
    def loadOptimizer(self):
       
            if "Adam" in self.cfg["optimizer"]["name"]:         
                op=optim.Adam(self.model.parameters(),**self.cfg["optimizer"]["optimizer_params"])
                if self.cfg["optimizer"]["scheduler"]=="CosineAnnealingLR":  
                
                    scheduler = CosineAnnealingLR(op,**self.cfg["optimizer"]["scheduler_params"])
                return op,scheduler
            
            # elif self.cfg.train.optimizer.name=="SGD":             
            #     op= optim.SGD(model.parameters(),lr=0.005, weight_decay=0.001)
            #     scheduler = StepLR(op, step_size=2, gamma=0.1,verbose=True)
            #     return op, scheduler
            # else:
            #     raise Exception (" not support that optimizer")
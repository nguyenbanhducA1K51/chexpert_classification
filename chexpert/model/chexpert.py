import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
# from ..datasets.data import 
# from ..model import modelUtils,backbone
import dataUtils
import modelUtils
import backbone
import Metric
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from torch.optim import Adam
import torch.optim as optim
import numpy as np
from enum import Enum
# from libauc.losses import AUCMLoss 
from libauc.optimizers import PESG 
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
class chexpertNet():
    def __init__(self,cfg,device,num_class):
 
        self.cfg=cfg
        self.disease=cfg.disease
        self.device=device
        self.num_class=num_class
        self.model=self.loadModel(self.cfg).to(device)
        self.optimizer, self.lr_scheduler=self.loadOptimizer(cfg,self.model)
        self.criterion=self.loadCriterion(cfg)
        self.metric=Metric.Metric(self.disease)
       
        
        
    def train_epoch(self,data_loader,epoch):
            log_interval=self.cfg.train.log_interval
            train_correct=0
            counter=0
            train_loss=0        
            self.model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                data=data.to(self.device).float()
                target=target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(y_score=output, y_true=target,device=self.device,beta=0.99).sum(1).mean(0) 
    
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                if batch_idx % log_interval == 0:
                    print(' Batch index {} Train Epoch: {} [{}/{} ({:.0f}%)]\t  Loss {: .5f}'.format( batch_idx,
                        epoch, batch_idx * len(data), len(data_loader.dataset), 
                        100. * batch_idx / len(data_loader),loss.item() ))
        
    def eval(self,data_loader):
        print ("VALIDATING :")
        self.model.eval()
        losses= Metric.AverageMeter()   
        y_score=[]
        y_true=[]
        with torch.no_grad():
            for data, target in data_loader:
                data=data.to(self.device).float()
                target=target.to(self.device)
                counter+=1
                y_true.append(target)
                output = self.model(data)
                y_score.append(output)
        
                loss = self.criterion(y_score=output, y_true=target,device=self.device,beta=0.99).sum(1).mean(0)    
                print ("loss: {: .5f}".format (loss.item()))   
                losses.update(loss.item())        
               
        y_score=torch.concat(y_score,dim=0).detach()
        y_true=torch.concat(y_true,dim=0).detach()

        metric=self.metric.compute_metrics(outputs=y_score,targets=y_true,losses=losses.mean)
        print (" Metric: {}".format(metric))
        return metric
        
    
    def train_epochs (self,train_loader,val_loader):
        save_best_model = modelUtils.SaveBestModel()
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
            self.train_epoch( data_loader=train_loader,epoch=epoch)
            self.lr_scheduler.step()
            
            metric=self.eval(data_loader=val_loader)          
            save_best_model(
               metric, epoch, self.model, self.optimizer, self.criterion,best

        )
        print('-'*100)
    def progressive_train_epochs(self,train_loader,val_loader,progress_train_loader,progress_test_loader):
        print(" Train on small size image set")
        for epoch in range(1, self.cfg.progressive_train.epochs + 1):
            
            print(f"[]: Epoch {epoch} of {self.cfg.progressive_train.epochs}")
            self.train_epoch( data_loader=progress_train_loader,epoch=epoch)
            
        
        self.eval(data_loader=progress_test_loader)  
        print(" Train on default size image set")
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
            self.train_epoch( data_loader=train_loader,epoch=epoch)
            
        self.eval(data_loader=val_loader)   


    def loadModel(self,cfg):
        if cfg.backbone.name=="densenet121":
            if cfg.train_mode=="default":
                return backbone.DenseNetClassifier(num_classes=self.num_class,pretrain= cfg.backbone.pretrain)
            elif cfg.train_mode=="progressive":
                return backbone.DenseNetClassifier(num_classes=self.num_class,pretrain=cfg.backbone.pretrain)
        elif cfg.backbone.name=="convnext_t":
            if cfg.train_mode.name=="default":
                return backbone.ConvNextClassifier(num_classes=self.num_class,pretrain=cfg.backbone.pretrain)
            elif cfg.train_mode.name=="progressive":
                return backbone.ConvNextClassifier(num_classes=self.num_class,pretrain=cfg.backbone.pretrain)

    def loadCriterion(self,cfg):
        if cfg.criterion=="bce":
            return nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        elif cfg.criterion=="balanceBCE":
            return dataUtils.balanceCE
        else:
            raise Exception (" not support that criterion")
       
            
    def loadOptimizer(self,cfg,model):
        if cfg.train.optimizer.name=="Adam":
        
            op=optim.Adam(model.parameters(),lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay)
            scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
            return op,scheduler
        elif cfg.train.optimizer.name=="SGD":
            
            op= optim.SGD(model.parameters(),lr=0.005, weight_decay=0.001)
            scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
            return op, scheduler
        else:
            raise Exception (" not support that optimizer")

       

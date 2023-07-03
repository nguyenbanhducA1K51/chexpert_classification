
import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
from datasets import dataUtils, dataset
from datasets.common import csv_index
from . import modelUtils,backbone
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from torch.optim import Adam
import torch.optim as optim
import numpy as np
from enum import Enum
from libauc.optimizers import PESG 
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import ttach as tta 
from tqdm import tqdm
class chexpertNet():
    def __init__(self,cfg,device):
        self.cfg=cfg
        self.device=device
        self.class_idx=cfg.class_idx
        self.classes=[]
        for idx in self.class_idx:
            self.classes.append(csv_index[str(idx)])
        self.model=self.loadModel().to(device)
        self.optimizer, self.lr_scheduler=self.loadOptimizer(self.model)
        self.criterion=self.loadCriterion()
        self.metric=modelUtils.Metric(self.classes)   
        self.train_loader,self.test_loader= dataset.loadData(cfg=cfg)      
        self.prog_optimizer,self.prog_lr_scheduler=self.loadOptimizer(self.model,mode="progressive")
        self.progress_train_loader,self.progress_test_loader=dataset.loadData(cfg=cfg,mode="progressive")   
        self.tta_test_loader=dataset.tta_loader(self.cfg)
        self.save_best_model=modelUtils.SaveBestModel(cfg=self.cfg)
    def train_epoch(self,data_loader,epoch,model,optim):
            log_interval=self.cfg.train.log_interval
            train_correct=0 
            outputs=[]
            targets=[]       
            losses= modelUtils.AverageMeter()       
            model.train()
            with tqdm(data_loader, unit="batch") as tepoch:              
                for batch_idx, (data, target) in enumerate(tepoch):
                    tepoch.set_description("Batch {}".format(batch_idx) )
                    data=data.to(self.device).float()
                    target=target.to(self.device)
                    optim.zero_grad()
                    output = model(data)   
                    targets.append(target)  
                    outputs.append(output)               
                    loss = self.criterion(output=output, target=target).sum(1).mean(0)       
                    losses.update(loss.item()) 
                    loss.backward()
                    optim.step()
                    if batch_idx % log_interval == 0:
                        tepoch.set_postfix ( loss=loss.item())
            outputs=torch.concat(outputs,dim=0).detach()
            targets=torch.concat(targets,dim=0).detach()
            metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
            return metric

    def eval(self,data_loader,model,epoch):
        print ("Epoch {} VALIDATING :".format(epoch))
        model.eval()
        losses= modelUtils.AverageMeter()   
        outputs=[]
        targets=[]
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for idx,(data, target) in enumerate(tepoch):
                    data=data.to(self.device).float()
                    target=target.to(self.device)              
                    targets.append(target)
                    output =model(data)
                    outputs.append(output)        
                    loss = self.criterion(output=output, target=target).sum(1).mean(0)    
                    losses.update(loss.item()) 
                    if idx%4==0:
                        tepoch.set_postfix(loss=loss.item())                                     
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()
        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))
        modelUtils.recordTraining(cfg=self.cfg,epoch=epoch,metric=metric)
        return metric      
    def eval_tta(self,model,data_loader,epoch):
        model.eval()   
        print ("Epoch {} VALIDATING TTA:".format(epoch))
        all_samples=len(data_loader.dataset)
        output_sum=torch.zeros(all_samples,len(self.class_idx)).to(self.device)
        out_targets=torch.zeros(all_samples,len(self.class_idx)).to(self.device)
        for i in range(self.cfg.tta.times):
            print ("validate time {}".format (i))
            losses= modelUtils.AverageMeter()   
            outputs=[]
            targets=[]
            with torch.no_grad():
                with tqdm(data_loader, unit="batch") as tepoch:
                        for idx, (data, target) in enumerate(tepoch):
                            data=data.to(self.device).float()
                            target=target.to(self.device) 
                            targets.append(target)
                            output = model(data)
                            outputs.append(output)        
                            loss = self.criterion(output=output, target=target).sum(1).mean(0)    
                            losses.update(loss.item()) 
                            if idx%4==0:
                                tepoch.set_postfix(loss=loss.item())                                     
                outputs=torch.concat(outputs,dim=0).detach()
                targets=torch.concat(targets,dim=0).detach()
                assert output_sum.shape==outputs.shape
                output_sum+=outputs
                out_targets=targets
        average_output=output_sum/self.cfg.tta.times
        metric=self.metric.compute_metrics(outputs=average_output,targets=out_targets)
        print (" Mean AUC using tta : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))
        modelUtils.recordTraining(cfg=self.cfg,epoch=epoch,metric=metric)
        return metric                           
    def train_epochs (self):    
        if self.cfg.load_ckp=="True":
            model=self.loadckpModel()
            for epoch in range(1, self.cfg.train.epochs + 1):               
                print("Evaluate checkpoint model")
                if self.cfg.tta.usetta=="False":
                    self.eval(data_loader=self.test_loader,model=model,epoch=epoch)  
                else:
                    self.eval_tta(model=model,epoch=epoch,data_loader=self.tta_test_loader)

        else:            
            for epoch in range(1, self.cfg.train.epochs + 1):
                print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
 
                self.train_epoch( data_loader=self.train_loader,epoch=epoch,model=self.model,optim=self.optimizer)              
                if self.cfg.tta.usetta=="False":
                    metric=self.eval(data_loader=self.test_loader,model=self.model,epoch=epoch)  
                else:
                    metric=self.eval_tta(model=self.model,epoch=epoch,data_loader=self.tta_test_loader)

                self.save_best_model(
                metric, epoch, self.model, self.optimizer, self.criterion)
                self.lr_scheduler.step()
        print ("Finish default training")
        print('-'*100)
    def progressive_train_epochs(self):
        print(" Train on small size image set")   
        for epoch in range(1, self.cfg.progressive_train.epochs + 1):
            print(f"[]: Epoch {epoch} of {self.cfg.progressive_train.epochs}")
            self.train_epoch( data_loader=self.train_loader,epoch=epoch,model=self.model,optim=self.prog_optimizer)     
            self.eval(data_loader=self.test_loader,model=self.model,epoch=epoch)  
            self.prog_lr_scheduler.step()        
        print(" Train on default size image set")
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
            self.train_epoch( data_loader=self.train_loader,epoch=epoch,model=self.model,optim=self.optimizer) 
            if self.cfg.tta.usetta=="False" : 
                metric=self.eval(data_loader=self.test_loader,model=self.model,epoch=epoch+int(self.cfg.progressive_train.epochs))                 
            else:   
                metric=self.eval_tta(model=self.model,epoch=epoch,data_loader=self.tta_test_loader) 
            self.save_best_model(
                metric, epoch, self.model, self.optimizer, self.criterion)
            self.lr_scheduler.step()   
        print ("Finish progressive training")

    def loadModel(self):
        if self.cfg.backbone.name=="densenet121":    
            if self.cfg.train_mode.name=="default":
                return backbone.DenseNetClassifier(num_classes=len(self.class_idx),pretrain= self.cfg.backbone.pretrain)
            elif self.cfg.train_mode.name=="progressive":
                return backbone.DenseNetClassifier(num_classes=len(self.class_idx),pretrain=self.cfg.backbone.pretrain)
        elif self.cfg.backbone.name=="convnext_t":
            if self.cfg.train_mode.name=="default":
                return backbone.ConvNextClassifier(num_classes=len(self.class_idx),pretrain=self.cfg.backbone.pretrain)
            elif self.cfg.train_mode.name=="progressive":
                return backbone.ConvNextClassifier(num_classes=len(self.class_idx),pretrain=self.cfg.backbone.pretrain)
    def loadckpModel(self):
        checkpoint=torch.load("output/best_model.pth")
        model_state_dict = checkpoint['model_state_dict']
        model=self.loadModel()
        model.load_state_dict(model_state_dict)
        return model
    def loadCriterion(self):
        if self.cfg.criterion=="bce":
            return nn.BCEWithLogitsLoss(reduction='none')
        elif self.cfg.criterion=="balanceBCE":
            return dataUtils.balanceBCE(beta=self.cfg.balanceBCE.beta,device=self.device)
        else:
            raise Exception (" not support that criterion")
                
    def loadOptimizer(self,model,mode="default"):
        if mode=="default":
            if self.cfg.train.optimizer.name=="Adam":         
                op=optim.Adam(model.parameters(),lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
                scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
                return op,scheduler
            elif self.cfg.train.optimizer.name=="SGD":             
                op= optim.SGD(model.parameters(),lr=0.005, weight_decay=0.001)
                scheduler = StepLR(op, step_size=2, gamma=0.1,verbose=True)
                return op, scheduler
            else:
                raise Exception (" not support that optimizer")
        elif mode=="progressive":
            if self.cfg.progressive_train.optimizer.name=="Adam":   
                op=optim.Adam(model.parameters(),lr=self.cfg.progressive_train.optimizer.lr, weight_decay=self.cfg.progressive_train.optimizer.weight_decay)
                scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
                return op,scheduler
            elif self.cfg.progressive_train.optimizer.name=="SGD":         
                op= optim.SGD(model.parameters(),lr=0.005, weight_decay=0.001)
                scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
                return op, scheduler
            else:
                raise Exception (" not support that optimizer")

       

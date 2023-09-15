
import sys
from . import utils
import torch
import json,os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import ast

from data.dataset import ChestDataset
from data.common import csv_index
# sys.path.append( str(Path(__file__).resolve().parent.parent) )
sys.path.append("../")
from model import backbone

from .Loader import  Loader
class chexpertNet():
    def __init__(self,cfg,device,fold):
        self.Loader=Loader(cfg,device)
        self.fold=fold
        self.cfg=cfg
        self.device=device
        # self.class_idx=ast.literal_eval(cfg["train_params"]["class_idx"])
        self.class_idx=cfg["train_params"]["class_idx"].split(",")
        self.classes=[]
        for idx in self.class_idx:
            self.classes.append(csv_index[str(idx)])

        self.model=self.Loader.model.to(device)

        self.optim, self.lr_scheduler=self.Loader.optim,self.Loader.scheduler

        self.loss=self.Loader.loss

        self.metric=utils.Metric(self.classes)  

        self.prog_optimizer,self.prog_lr_scheduler=self.Loader.prog_optim,self.Loader.prog_scheduler

        
    def train_epoch(self,data_loader,epoch,model,optim):
            log_interval=self.cfg.train.log_interval
            outputs=[]
            targets=[]       
            losses= utils.AverageMeter()       
            model.train()
            with tqdm(data_loader, unit="batch") as tepoch:              
                for batch_idx, (data, target) in enumerate(tepoch):
                    tepoch.set_description("Batch {}".format(batch_idx) )
                    data=data.to(self.device).float()
                    target=target.to(self.device).float()
                    optim.zero_grad()
                    output = model(data)   
                    targets.append(target)  
                    outputs.append(output)               
                    loss = self.loss(output, target).sum(1).mean(0)       
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
        losses= utils.AverageMeter()   
        outputs=[]
        targets=[]
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for idx,(data, target) in enumerate(tepoch):
                    data=data.to(self.device).float()
                    target=target.to(self.device).float()          
                    targets.append(target)
                    output =model(data)
                    outputs.append(output)        
                    loss = self.loss(output, target).sum(1).mean(0)    
                    losses.update(loss.item()) 
                    if idx%4==0:
                        tepoch.set_postfix(loss=loss.item())                                     
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()
        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))

        return metric      
    
    def train_epochs (self,train_loader,val_loader,lowres_train_loader=None,fold=1):    
        train_metrics=[]
        val_metrics=[]
               
        if lowres_train_loader is not None :
            print(" TRAIN ON LOW-RES DATASET")     
            for epoch in range(1, self.cfg["epochs"] + 1):
                print(f"[]:  Epoch {epoch} of {self.cfg['epochs']} on Low-res dataset")
                self.train_epoch( lowres_train_loader,epoch,self.model,self.prog_optimizer)  
                self.prog_lr_scheduler.step()

        print(" TRAIN ON HI-RES DATASET")  
        if self.cfg.train.early_stopping.use=="True":
            best_mean_AUC=- float('inf')
            epochs_no_improve = 0
            for epoch in range(1,self.cfg.train.epochs+1):
                train_metric=self.train_epoch( data_loader=train_loader,epoch=epoch,model=self.model,optim=self.optimizer)              
                train_metrics.append(train_metric)
                
                val_metric=self.eval(data_loader=val_loader,model=self.model,epoch=epoch)  
                best_mean_AUC=max(best_mean_AUC,val_metric["meanAUC"])
                if val_metric["meanAUC"] == best_mean_AUC:
                    epochs_no_improve = 0
                elif val_metric["meanAUC"]< best_mean_AUC:
                    epochs_no_improve +=1
                if epochs_no_improve == self.cfg["train"]["patient"]:
                    print (f"Early stop on epoch {epoch}")
                    break                  
                val_metrics.append(val_metric)
        else:
            for epoch in range(1,self.cfg.train.epochs+1):
                train_metric=self.train_epoch( train_loader,epoch,self.model,self.optimizer,self.lr_scheduler)              
                train_metrics.append(train_metric)       
                val_metric=self.eval(data_loader=val_loader,model=self.model,epoch=epoch)  
                best_mean_AUC=max(best_mean_AUC,val_metric["meanAUC"])
                val_metrics.append(val_metric)          
                self.lr_scheduler.step()
   
        print ("Finish default training")
        print('-'*100)
        return train_metrics ,val_metrics 

    def train(self):

        train_dataset=ChestDataset(cfg=self.cfg,mode="train",fold=self.fold)
    
        lowres_train_dataset=ChestDataset(cfg=self.cfg,mode="train", fold=self.fold,train_mode="progressive") if self.cfg.train_mode.name=="progressive" else None
        print(f'FOLD {self.fold}')
        print('--------------------------------')          
        train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=self.cfg["train_params"]["batch_size"])
        val_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.cfg.train.batch_size)
        lowres_train_loader=torch.utils.data.DataLoader(
                    lowres_train_dataset, 
                    batch_size=self.cfg["train_params"]["batch_size"]) if lowres_train_dataset is not None else None
        
        train_metrics,val_metrics=self.train_epochs(train_loader,val_loader,lowres_train_loader,self.fold)

        
        utils.save_metrics_and_models({"train_stats":train_metrics,"val_stats":val_metrics},self.model,self.fold)

 


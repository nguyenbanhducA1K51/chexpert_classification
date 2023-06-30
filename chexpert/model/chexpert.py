
import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
from datasets import dataUtils, dataset
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
        self.disease=cfg.disease
        self.device=device
        self.num_class=len(cfg.disease)
        self.model=self.loadModel().to(device)
        # self.ttamodel = tta.ClassificationTTAWrapper(model,
        self.optimizer, self.lr_scheduler=self.loadOptimizer(self.model)
        self.criterion=self.loadCriterion()
        self.metric=modelUtils.Metric(self.disease)   
        self.train_data,self.test_data= dataset.loadData(cfg=cfg)
        if cfg.train_mode=="progressive":
            self.prog_optimizer,self.prog_lr_scheduler=self.loadOptimizer(self.model,mode="progressive")
    def train_epoch(self,data_loader,epoch,model,optim):
            log_interval=self.cfg.train.log_interval
            train_correct=0
           
            train_loss=0        
            model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
        
                data=data.to(self.device).float()
                target=target.to(self.device)
                optim.zero_grad()
                output = self.model(data)
                
                loss = self.criterion(output=output, target=target).sum(1).mean(0) 
    
                loss.backward()
                optim.step()
                # self.scheduler.step()
                if batch_idx % log_interval == 0:
                    print(' Batch index {} Train Epoch: {} [{}/{} ({:.0f}%)]\t  Loss {: .5f}'.format( batch_idx,
                        epoch, batch_idx * len(data), len(data_loader.dataset), 
                        100. * batch_idx / len(data_loader),loss.item() ))
        
    def eval(self,data_loader,model,epoch):
        print ("VALIDATING :")
        model.eval()
        losses= modelUtils.AverageMeter()   
        outputs=[]
        targets=[]
        with torch.no_grad():
            for data, target in data_loader:
                data=data.to(self.device).float()
                target=target.to(self.device)              
                targets.append(target)
                output = self.model(data)
                outputs.append(output)        
                loss = self.criterion(output=output, target=target).sum(1).mean(0)    
                print ("loss: {: .5f}".format (loss.item()))   
                losses.update(loss.item())        
               
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()

        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))
        modelUtils.recordTraining(cfg=self.cfg,epoch=epoch,metric=metric)
        return metric
        
    # def eval_tta(self,data_loader,model,epoch):
    #     print ("VALIDATING TTA:")
    #     with torch.no_grad():
    #         with tqdm(dataLoader, unit="batch") as tepoch:
    #                 for idx, (_,x,y_true) in enumerate(tepoch):
    #                     for times in self.cfg.tta.times:
    #                         y_aug=dataset.tta_transform(x.clone() )
    #                         y_pred=

    def train_epochs (self):
        save_best_model = modelUtils.SaveBestModel()
        if self.cfg.load_ckp=="True":
            testLoader = torch.utils.data.DataLoader(self.testdata, batch_size=1)                     
            model=self.loadckpModel()
            print("Evaluate checkpoint model")
            metric=self.eval(data_loader=testLoader,model=model)  
        else:            
            for epoch in range(1, self.cfg.train.epochs + 1):
                print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
                trainLoader=torch.utils.data.DataLoader (self.train_data,batch_size=self.cfg.train.batch_size,shuffle=True)
                testLoader = torch.utils.data.DataLoader(self.test_data, batch_size=1,shuffle=False)  
                self.train_epoch( data_loader=trainLoader,epoch=epoch,model=self.model,optim=self.optimizer)              
                metric=self.eval(data_loader=testLoader,model=self.model,epoch=epoch)  
                save_best_model(
                metric, epoch, self.model, self.optimizer, self.criterion)
                self.lr_scheduler.step()
        print ("Finish default training")
        print('-'*100)
    def progressive_train_epochs(self):
        print(" Train on small size image set")
        optimizer,lr_scheduler=self.loadOptimizer(cfg=self.cfg,mode="progressive")
        for epoch in range(1, self.cfg.progressive_train.epochs + 1):
            train_data, test_data =dataset.loadData(cfg=self.cfg,mode="progressive")
            train_loader=torch.utils.data.DataLoader (train_data,batch_size=self.cfg.progressive_train.batch_size,shuffle=True)
            test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=False)
            print(f"[]: Epoch {epoch} of {self.cfg.progressive_train.epochs}")
            self.train_epoch( data_loader=train_loader,epoch=epoch,model=self.model,optim=optimizer)
            self.eval(data_loader=test_loader,model=self.model,epoch=epoch)  
            lr_scheduler.step()
              
        print(" Train on default size image set")
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loader=torch.utils.data.DataLoader (self.train_data,batch_size=self.cfg.train.batch_size,model=self.model,shuffle=True)
            test_loader=torch.utils.data.DataLoader(self.test_data,batch_size=1,shuffle=False)
            print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
            self.train_epoch( data_loader=train_loader,epoch=epoch,model=self.model,optim=self.optimizer)   
            self.eval(data_loader=val_loader,model=self.model,epoch=epoch+int(self.cfg.progressive_train.epochs))   
            self.lr_scheduler.step()
        print ("Finish progressive training")

    def loadModel(self):
        if self.cfg.backbone.name=="densenet121":    
            if self.cfg.train_mode.name=="default":
                return backbone.DenseNetClassifier(num_classes=self.num_class,pretrain= self.cfg.backbone.pretrain)
            elif self.cfg.train_mode.name=="progressive":
                return backbone.DenseNetClassifier(num_classes=self.num_class,pretrain=self.cfg.backbone.pretrain)
        elif self.cfg.backbone.name=="convnext_t":
            if self.cfg.train_mode.name=="default":
                return backbone.ConvNextClassifier(num_classes=self.num_class,pretrain=self.cfg.backbone.pretrain)
            elif self.cfg.train_mode.name=="progressive":
                return backbone.ConvNextClassifier(num_classes=self.num_class,pretrain=self.cfg.backbone.pretrain)
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

       

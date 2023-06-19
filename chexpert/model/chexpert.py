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
        self.optimizer, self.scheduler=self.loadOptimizer(cfg,self.model)
        self.criterion=self.loadCriterion(cfg)
        
    def train_epoch(self,data_loader,epoch):
            # epochs=self.cfg.train.epochs
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
                loss = self.criterion(y_score=output, y_true=target,device=self.device,beta=0.5).sum(1).mean(0) 
    
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
       
        counter = 0
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
        
                loss = self.criterion(y_score=output, y_true=target,device=self.device,beta=0.5).sum(1).mean(0)    
                print ("loss: {: .5f}".format (loss.item()))           
               
        y_score=torch.concat(y_score,dim=0).detach()
        y_true=torch.concat(y_true,dim=0).detach()

        AUC=calculateAUC(y_score=y_score,y_true=y_true,disease=self.disease)
        print ("Val set : AUC : {}".format (AUC))

    
    def train_epochs (self,train_data,val_data):
        save_best_model = modelUtils.SaveBestModel()
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
            self.train_epoch( data_loader=train_data,epoch=epoch)
            
            self.eval(data_loader=val_data)
           
        #     save_best_model(
        #         valid_epoch_loss, epoch, self.model, self.optimizer, self.criterion
        # )
        print('-'*50)
        
    def test(self,test_data):
        model=self.model
        if self.cfg.load_ckp:
            
            model.load_state_dict(torch.load("/root/project/chexpert/model/output/best_model.pth")['model_state_dict'])
        else:
            self.eval(test_data)
    def loadModel(self,cfg):
        if cfg.model=="densenet121":
            return backbone.DenseNetClassifier(num_classes=self.num_class)
    def loadCriterion(self,cfg):
            # return AUCM_MultiLabel(num_classes=14)
            # return AUCMLoss()
        return dataUtils.balanceCE
        # return nn.BCEWithLogitsLoss(reduction='none').to(self.device)
            
    def loadOptimizer(self,cfg,model):
        if cfg.train.optimizer.name=="Adam":
        
            op=optim.Adam(model.parameters(),lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay)
            scheduler = StepLR(op, step_size=30, gamma=0.1)
            return op,scheduler
        elif cfg.train.optimizer.name=="SGD":
            
            op= optim.SGD(model.parameters(),lr=0.005, weight_decay=0.001)
            scheduler = StepLR(op, step_size=30, gamma=0.1)
            return op, scheduler
       

def compare(output,target):
    output=torch.sigmoid(output)
    output[output>=0.5]=1.
    output[output<0.5]=0
    return output.eq(target.view_as(output)).sum()

def calculateAUC (y_score,y_true,disease):
    # y_score , y_true are tensor of shape N* (num class)
    y_score=torch.sigmoid(y_score)
    y_score=y_score.cpu().detach().numpy()
    y_true=y_true.cpu().detach().numpy()
    AUC=[]
    for i in range (y_score.shape[1]):
        y_t=y_true[:,i].copy()
        y_s=y_score[:,i].copy()
        if len(np.unique(y_t )) !=2:
            # print ("only one class present in disease "+str(i))
            continue         
        else:
            score =roc_auc_score(y_true=y_true[:,i].copy(),y_score=y_score[:,i].copy()) 
            score=round(score, 2)
            AUC.append ( ("class {}".format(disease[i]),score))      
    return AUC
def compute_metrics(outputs, targets, losses,disease):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:,i], outputs[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics
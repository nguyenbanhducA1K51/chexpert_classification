import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
import data
import modelUtils
import backbone
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
import torch.optim as optim
import numpy as np
from enum import Enum

disease= { 

    "0":"No Finding",
    "1":"Enlarged Cardiomediastinum",
    "2":'Cardiomegaly',
    "3":'Lung Opacity',
    "4":'Lung Lesion',
    "5":'Edema',
    "6":'Consolidation',
    "7":'Pneumonia',
    "8":"Atelectasis",
    "9": "Pneumothorax",
    "10":"Pleural Effusion",
    "11": "Pleural Other",
    "12": "Fracture",
    "13": "Support Devices" 
}

class chexpertNet():
    def __init__(self,cfg,device):
 
        self.cfg=cfg
        self.device=device
        self.num_class=cfg.num_class
        self.model=loadModel(self.cfg).to(device)
        self.optimizer=loadOptimizer(cfg,self.model)
        self.criterion=loadCriterion(cfg)
        
    def train_epoch(self,data_loader,epoch):
            # epochs=self.cfg.train.epochs
            log_interval=self.cfg.train.log_interval

            train_correct=0
            counter=0
            train_loss=0

           
            self.model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                data=data.to(self.device)
                target=target.to(self.device)
                counter+=1
                self.optimizer.zero_grad()
                output = self.model(data)
            
                train_correct+=compare(output=output,target=target)
                loss = self.criterion(output, target)
                train_loss+=loss.item()
                loss.backward()
                self.optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset), 
                        100. * batch_idx / len(data_loader),  loss.item()))
            print ("Train accuracy : {:.0f}% ".format (100. *train_correct/(len(data_loader.dataset)*self.num_class)))
            epoch_loss=  train_loss/counter
            epoch_acc=100. *train_correct/(len(data_loader.dataset)*self.num_class)
            
            
            return epoch_loss, epoch_acc
    def eval(self,data_loader):
        print ("VALIDATING :")
        self.model.eval()
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        y_score=[]
        y_true=[]
        with torch.no_grad():
            for data, target in data_loader:
                data=data.to(self.device)
                target=target.to(self.device)
                counter+=1
                y_true.append(target)

                output = self.model(data)
                y_score.append(output)
                loss = self.criterion(output, target)
                valid_running_loss += loss.item()
                valid_running_correct += compare(output=output,target=target)
        
        print('\nEval set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_running_correct, len(data_loader.dataset)*self.num_class,
        100. * valid_running_correct / (len(data_loader.dataset)*self.num_class) ))
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / (len(data_loader.dataset)*self.num_class ))
        y_score=torch.concat(y_score,dim=0).detach()
        y_true=torch.concat(y_true,dim=0).detach()

        AUC=calculateAUC(y_score=y_score,y_true=y_true)
        print ("AUC : {}".format (AUC))
        return epoch_loss, epoch_acc
    
    def train_epochs (self,train_data,val_data):
        save_best_model = modelUtils.SaveBestModel()
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"[INFO]: Epoch {epoch} of {self.cfg.train.epochs}")
            train_epoch_loss, train_epoch_acc =  self.train_epoch( data_loader=train_data,epoch=epoch)
            
            valid_epoch_loss, valid_epoch_acc =self.eval(data_loader=val_data)
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            save_best_model(
                valid_epoch_loss, epoch, self.model, self.optimizer, self.criterion
        )
        print('-'*50)
        # print (" train accu {} \n val accu {}".format(train_acc,valid_acc))
        modelUtils.save_plots(train_acc, valid_acc, train_loss, valid_loss)


    def test(self,test_data):
        model=self.model
        if self.cfg.load_ckp:
            
            model.load_state_dict(torch.load("/root/project/chexpert/model/output/best_model.pth")['model_state_dict'])
        else:
            self.eval(test_data)
            
def loadOptimizer(cfg,model):
        if cfg.train.optimizer.name=="Adam":
        
            return optim.Adam(model.parameters(),lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay)
   
def loadModel(cfg):
        if cfg.model=="densenet121":
            return backbone.DenseNetClassifier(num_classes=cfg.num_class)
def loadCriterion(cfg):
        return F.binary_cross_entropy_with_logits
def compare(output,target):
    output=torch.sigmoid(output)
    output[output>=0.5]=1.
    output[output<0.5]=0
    return output.eq(target.view_as(output)).sum()

def calculateAUC (y_score,y_true):
    # y_score , y_true are tensor of shape N* (num class)
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
            # print ("unique {} {}".format (np.unique(y_t ), i))
            score =roc_auc_score(y_true[:,i].copy(),y_score[:,i].copy()) 
            score=round(score, 2)
            AUC.append ( ("class {}".format(disease[str(i)]),score))      
    return AUC
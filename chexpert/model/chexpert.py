
import sys
import torch
import json,os
from data.dataset import ChestDataset
from data import dataUtils
from data.common import csv_index
from . import modelUtils,backbone
from torch.nn import functional as F
from sklearn.model_selection import KFold
from torch.optim import Adam
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
import glob

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
        self.prog_optimizer,self.prog_lr_scheduler=self.loadOptimizer(self.model,mode="progressive")
        self.save_best_model=modelUtils.SaveBestModel(cfg=self.cfg)
    def train_epoch(self,data_loader,epoch,model,optim):
            log_interval=self.cfg.train.log_interval
            outputs=[]
            targets=[]       
            losses= modelUtils.AverageMeter()       
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
                    loss = self.criterion(output, target).sum(1).mean(0)       
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
                    target=target.to(self.device).float()          
                    targets.append(target)
                    output =model(data)
                    outputs.append(output)        
                    loss = self.criterion(output, target).sum(1).mean(0)    
                    losses.update(loss.item()) 
                    if idx%4==0:
                        tepoch.set_postfix(loss=loss.item())                                     
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()
        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))

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
                            loss = self.criterion(output, target).sum(1).mean(0)    
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
    def train_epochs (self,train_loader,val_loader,lowres_train_loader=None,fold=1):    
        train_metrics=[]
        val_metrics=[]
        if self.cfg.load_ckp=="True":
            model=self.loadckpModel()
            for epoch in range(1, self.cfg.train.epochs + 1):               
                print("Evaluate checkpoint model")
                val_metric=self.eval(data_loader=val_loader,model=model,epoch=epoch)  

        else:          
            if lowres_train_loader is not None :
                print(" TRAIN ON LOW-RES DATASET")         
                for epoch in range(1, self.cfg.progressive_train.epochs + 1):
                    print(f"[]:  Epoch {epoch} of {self.cfg.progressive_train.epochs} on Low-res dataset")
                    self.train_epoch( data_loader=lowres_train_loader,epoch=epoch,model=self.model,optim=self.prog_optimizer)  
             
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
                    if epochs_no_improve == self.cfg.train.early_stopping.patient:
                        print (f"Early stop on epoch {epoch}")
                        break                  
                    val_metrics.append(val_metric)
            else:
                for epoch in range(1,self.cfg.train.epochs+1):
                    train_metric=self.train_epoch( data_loader=train_loader,epoch=epoch,model=self.model,optim=self.optimizer)              
                    train_metrics.append(train_metric)
                    
                    val_metric=self.eval(data_loader=val_loader,model=self.model,epoch=epoch)  
                    best_mean_AUC=max(best_mean_AUC,val_metric["meanAUC"])
                    val_metrics.append(val_metric)          
                self.lr_scheduler.step()
   
        print ("Finish default training")
        print('-'*100)
        return train_metrics ,val_metrics
    
    def test(self):
        print(f"EVALUATING ON TEST SET")
        test_dataset= ChestDataset(cfg=self.cfg,mode="test") 
        data_loader=torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size=self.cfg.train.batch_size)
        current_file_path = os.path.abspath(__file__)

        parent_directory = os.path.dirname(current_file_path)
        models_save_path=os.path.join (parent_directory, "output/models")
        
        files = [f for f in os.listdir(models_save_path) if os.path.isfile(os.path.join(models_save_path, f))]
        pth_files = [f for f in files if f.endswith('.pth')]
        models=[]
        for file in pth_files:
            path=os.path.join(models_save_path,file)
            state_dict=torch.load(path)
            new_model=copy.deepcopy(self.model)
            new_model.load_state_dict(state_dict['model_state_dict'])
            models.append(new_model)
        for model in models:
            model.to(self.device)
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
                    tmp_output =[model(data) for model in models]
                    output=torch.mean(torch.stack(tmp_output), dim=0)
                    outputs.append(output)        
                    loss = self.criterion(output, target).sum(1).mean(0)    
                    losses.update(loss.item()) 
                    if idx%4==0:
                        tepoch.set_postfix(loss=loss.item())                                     
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()
        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))  

    

    def k_fold_train(self):

        k_folds = self.cfg.k_fold
        kfold = KFold(n_splits=k_folds, shuffle=True)
        train_dataset=ChestDataset(cfg=self.cfg)
    
        lowres_train_dataset=ChestDataset(cfg=self.cfg,train_mode="progressive") if self.cfg.train_mode.name=="progressive" else None

        multiple_train_metrics=[]
        multiple_val_metrics=[]
        self.models=self.loadModel()
        models=[]

        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
            print(f'FOLD {fold+1}')
            print('--------------------------------')          
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=self.cfg.train.batch_size, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=self.cfg.train.batch_size, sampler=val_subsampler,)
            lowres_train_loader=torch.utils.data.DataLoader(
                        lowres_train_dataset, 
                        batch_size=self.cfg.train.batch_size, sampler=train_subsampler) if self.cfg.train_mode.name=="progressive" else None

            
            self.model.apply(reset_weights)
            train_metrics,val_metrics=self.train_epochs(train_loader,val_loader,lowres_train_loader,fold)
            multiple_train_metrics.append(train_metrics)
            multiple_val_metrics.append(val_metrics)
            models.append(copy.deepcopy(self.model.state_dict()))

            mean_aucs_of_epochs=np.mean([data["meanAUC"] for data in val_metrics]),
            highest_mean_auc=np.max([data["meanAUC"] for data in val_metrics] ),
        print (f"Finish train on fold {fold+1} with mean auc of epochs {mean_aucs_of_epochs}, highes mean auc {highest_mean_auc}")
        modelUtils.save_metrics_and_models({"train_stats":multiple_train_metrics,"val_stats":multiple_val_metrics},models)

    def switch_train_test(self) :
        # I observe that there is a significant difference between train loss and val/test loss (~ 3 for val/test loss and ~0.2 for train loss). I suspect that this is overfitting,
        # so i try out swaping train and test set, which mean training on test set and test on train set 
        # when run this method, val loss and train loss are almost similar, so overfit seem to be thw answer
        train_dataset=ChestDataset(cfg=self.cfg,mode='test')
        test_dataset=ChestDataset(cfg=self.cfg)
        train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=self.cfg.train.batch_size)
        test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.cfg.train.batch_size)
        self.train_epochs(train_loader,test_loader)
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
       
            if self.cfg.train.optimizer.name=="Adam":         
                op=optim.Adam(model.parameters(),lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
                if self.cfg.train.optimizer.scheduler=="CosineAnnealingLR":  
                    t_max=     self.cfg.train.optimizer.scheduler_params.T_max 
                    eta_min=self.cfg.train.optimizer.scheduler_params.eta_min
                    scheduler = CosineAnnealingLR(op,t_max,eta_min, verbose=True)
                return op,scheduler
            elif self.cfg.train.optimizer.name=="SGD":             
                op= optim.SGD(model.parameters(),lr=0.005, weight_decay=0.001)
                scheduler = StepLR(op, step_size=2, gamma=0.1,verbose=True)
                return op, scheduler
            else:
                raise Exception (" not support that optimizer")


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

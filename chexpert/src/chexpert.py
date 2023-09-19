
import sys

import torch
import json,os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import ast
import copy

# sys.path.append( str(Path(__file__).resolve().parent.parent) )
sys.path.append("../")
sys.path.append("/root/repo/chexpert_classification/chexpert/src")
sys.path.append("/root/repo/chexpert_classification/chexpert")
from model import backbone
import utils
from data.dataset import ChestDataset
from data.common import csv_index
from loader import  Loader
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

        # f and l are abbrev for frontal and lateral
        self.f_model, self.l_model=self.Loader.f_model.to(device), self.Loader.l_model.to(device)

        (self.f_optim, self.f_lr_scheduler), ( self.l_optim, self.l_lr_scheduler)=self.Loader.optim,self.Loader.scheduler

        self.loss=self.Loader.loss

        self.metric=utils.Metric(self.classes)  

        self.prog_optimizer,self.prog_lr_scheduler=self.Loader.prog_optim,self.Loader.prog_scheduler

        
    def train_epoch(self,dict, epoch):
            # f_data_loader,l_data_loader,f_model,l_model, f_optim,l_optim
        
            combine_outputs=[]
            combine_targets=[] 

            combine_loss=utils.AverageMeter()  

            for k, v in dict.items():
                print (f"{k} train epoch{epoch}")
                model=dict[k][0]
                data_loader=dict[k][1]



                optim=dict[k][2]
                log_interval=self.cfg["train_params"]["log_interval"]
                outputs=[]
                targets=[]       
                losses= utils.AverageMeter()       
                model.train()
                with tqdm(data_loader, unit="batch") as tepoch:              
                    for batch_idx, (data, target,_,_) in enumerate(tepoch):
                        tepoch.set_description("Batch {}".format(batch_idx) )
                        data=data.to(self.device).float()
                        target=target.to(self.device).float()
                        optim.zero_grad()
                        output = model(data)   
                        targets.append(target)  
                        outputs.append(output)  

                        combine_targets.append(target)  
                        combine_outputs.append(output)  


                        loss = self.loss(output, target).sum(1).mean(0)       
                        losses.update(loss.item()) 
                        combine_loss.update(loss.item())

                        loss.backward()

                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
                        optim.step()                     
                        if batch_idx % log_interval == 0:
                            tepoch.set_postfix ( loss=loss.item(),grad=grad_norm.item())
                outputs=torch.concat(outputs,dim=0).detach()
                targets=torch.concat(targets,dim=0).detach()
                metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
                print ( f" Mean AUC of {k} : {metric['meanAUC']: .3f}. AUC for each class: {metric['aucs']}")

            combine_outputs=torch.concat(combine_outputs,dim=0).detach()
            combine_targets=torch.concat(combine_targets,dim=0).detach()
            avg_metric= self.metric.compute_metrics(outputs=combine_outputs,targets=combine_targets,losses=combine_loss.mean)
            print ( f" Average AUC : {avg_metric['meanAUC']: .3f}. AUC for each class: {avg_metric['aucs']}")

            return avg_metric

    def eval(self,dict,mode="val"):
        metrics=[]
        combine_outputs=[]
        combine_targets=[] 

        combine_loss=utils.AverageMeter()  
        for k, v in dict.items():
            print (f"{k} {mode} ")
            model=dict[k][0]
            data_loader=dict[k][1]



            model.eval()
            losses= utils.AverageMeter()   
            outputs=[]
            targets=[]
            with torch.no_grad():
                with tqdm(data_loader, unit="batch") as tepoch:
                    for idx,(data, target,_,_) in enumerate(tepoch):
                        data=data.to(self.device).float()
                        target=target.to(self.device).float()          
                        targets.append(target)
                        output =model(data)
                        outputs.append(output) 

                        combine_targets.append(target)  
                        combine_outputs.append(output)   

                        loss = self.loss(output, target).sum(1).mean(0)    
                        losses.update(loss.item()) 
                        combine_loss.update(loss.item())
                        if idx%4==0:
                            tepoch.set_postfix(loss=loss.item())                                     
            outputs=torch.concat(outputs,dim=0).detach()
            targets=torch.concat(targets,dim=0).detach()
            metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
            print ( f" Mean AUC of {k} : {metric['meanAUC']: .3f}. AUC for each class: {metric['aucs']}")

        combine_outputs=torch.concat(combine_outputs,dim=0).detach()
        combine_targets=torch.concat(combine_targets,dim=0).detach()
        avg_metric= self.metric.compute_metrics(outputs=combine_outputs,targets=combine_targets,losses=combine_loss.mean)
        print ( f" Average AUC : {avg_metric['meanAUC']: .3f}. AUC for each class: {avg_metric['aucs']}")
        return avg_metric
           
    
    def train_epochs (self,f_train_loader,l_train_loader,f_val_loader,l_val_loader,f_test_loader,l_test_loader,fold=1):    
        
        train_metrics=[]
        val_metrics=[]
            
        print(" TRAIN ON HI-RES DATASET")  
        # if self.cfg ["train_params"]["early_stopping"]=="True":
        #     best_mean_AUC=- float('inf')
        #     epochs_no_improve = 0
        for epoch in range(1,self.cfg["epoch"]+1):
            avg_train=self.train_epoch(  {"frontal":[self.f_model,f_train_loader,self.f_optim] , "lateral":[self.l_model,l_train_loader,self.l_optim ] },epoch)


            train_metrics.append(avg_train)
            
            avg_val=self.eval( {"frontal":[self.f_model,f_val_loader] , "lateral":[self.l_model,l_val_loader] }) 
                                 
            val_metrics.append( avg_val)
            test_metrics=self.eval(  {"frontal":[self.f_model,f_test_loader] , "lateral":[self.l_model,l_test_loader] },mode="test")
                
            self.f_lr_scheduler.step()
            self.l_lr_scheduler.step()
   
        print ("Finish default training")
        print('-'*100)
        return train_metrics ,val_metrics 

    def train(self):

        # here for variable name, f is for frontal, l is for latera;
        f_trainset=ChestDataset(cfg=self.cfg,mode="train",fold=self.fold)
    
        l_trainset=ChestDataset(cfg=self.cfg,mode="train",fold=self.fold,view="lateral")
        
        f_lowres_trainset= ChestDataset(cfg=self.cfg,mode="train",train_mode="progressive",fold=self.fold)
        
        l_lowres_trainset= ChestDataset(cfg=self.cfg,mode="train",train_mode="progressive",fold=self.fold,view="lateral")
        
        f_valset=ChestDataset(cfg=self.cfg,mode="val",fold=self.fold)
    
        l_valset=ChestDataset(cfg=self.cfg,mode="val",fold=self.fold,view="lateral")
        
        f_testset= ChestDataset(cfg=self.cfg,mode="test")
        l_testset= ChestDataset(cfg=self.cfg,mode="test",view="lateral")


        print(f'FOLD {self.fold}')
        print('--------------------------------')          
        f_train_loader = torch.utils.data.DataLoader(
                    f_trainset, 
                    batch_size=self.cfg["train_params"]["batch_size"])
        
        l_train_loader= torch.utils.data.DataLoader(l_trainset,batch_size=self.cfg["train_params"]["batch_size"])
       
        f_val_loader = torch.utils.data.DataLoader(
                    f_valset, 
                    batch_size=self.cfg["train_params"]["batch_size"])
        
        l_val_loader= torch.utils.data.DataLoader(
            l_valset,batch_size=self.cfg["train_params"]["batch_size"])
       
        f_test_loader=torch.utils.data.DataLoader(
            f_testset,batch_size=1)
        l_test_loader=torch.utils.data.DataLoader(
            l_testset,batch_size=1)
        
        
        f_lowres_train_loader=torch.utils.data.DataLoader(
                    f_lowres_trainset, 
                    batch_size=self.cfg["train_params"]["batch_size"]) if  f_lowres_trainset is not None else None
        
        l_lowres_train_loader=torch.utils.data.DataLoader(
                    l_lowres_trainset, 
                    batch_size=self.cfg["train_params"]["batch_size"]) if  l_lowres_trainset is not None else None
        

        train_metrics,val_metrics=self.train_epochs(f_train_loader,l_train_loader,f_val_loader,l_val_loader,f_test_loader,l_test_loader, self.fold)

        
        utils.save_metrics_and_models({"train_stats":train_metrics,"val_stats":val_metrics},[self.f_model,self.l_model],self.fold)

    def test(self):
        print(f"EVALUATING ON TEST SET")
        test_dataset= ChestDataset(cfg=self.cfg,mode="test") 
        data_loader=torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size=self.cfg["train_params"]["batch_size"])
        
        models_save_path=Path(__file__).resolve().parent.parent/"output/models"
        
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
        losses= utils.AverageMeter()   
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
                    loss = self.loss(output, target).sum(1).mean(0)    
                    losses.update(loss.item()) 
                    if idx%4==0:
                        tepoch.set_postfix(loss=loss.item())                                     
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()
        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))  

    def train_frontal(self):
        train_dataset=ChestDataset(cfg=self.cfg,mode="train",fold=self.fold)
        train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=self.cfg["train_params"]["batch_size"])
        val_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.cfg["train_params"]["batch_size"])
        

if __name__=="__main__":
     
    import yaml
    argconfig="/root/repo/chexpert_classification/chexpert/config/config.yaml"
    with open(argconfig, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    net=chexpertNet(cfg=config,device="cuda",fold=1)
    net.train()


#  This file is for training a frontal/lateral classification if it is not specified in the data
import sys
import torch
import json,os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import ast
import copy
import random
import matplotlib.pyplot as plt

from data.dataset import ChestDataset
from data.common import csv_index
sys.path.append("../")
# sys.path.append("/root/repo/chexpert_classification/chexpert/")
# sys.path.append("/root/repo/chexpert_classification/chexpert/src")
from model import backbone
from model.frontal_cls import frontal_cls

import utils
from loader import  Loader
class chexpertNet():
    def __init__(self,cfg,device,fold):
       
        self.Loader=Loader(cfg,device)
        self.fold=fold
        self.cfg=cfg
        self.device=device

        self.class_idx=cfg["train_params"]["class_idx"].split(",")
        self.classes=[]
        for idx in self.class_idx:
            self.classes.append(csv_index[str(idx)])
        self.model_frontal,self.model_lateral=self.Loader.model.to(device)
        self.optim, self.lr_scheduler=self.Loader.optim,self.Loader.scheduler
        self.loss=self.Loader.loss
        self.metric=utils.Metric(self.classes)  

        self.prog_optimizer,self.prog_lr_scheduler=self.Loader.prog_optim,self.Loader.prog_scheduler

   

        
    def train_epoch(self,data_loader,epoch,model,optim):
            nums_visual=10
            prob=0.1
            visual_ls=[]

            log_interval=self.cfg["train_params"]["log_interval"]
            outputs=[]
            targets=[]       
            losses= utils.AverageMeter()       
            model.train()
            with tqdm(data_loader, unit="batch") as tepoch:              
                for batch_idx, (data, _,target,_) in enumerate(tepoch):
                    # print (type(target))
                    tepoch.set_description("Batch {}".format(batch_idx) )
                    data=data.to(self.device).float()
                    target=target.unsqueeze(1).to(self.device).float()
                    # target=target.to(self.device).float()
                   
                    
                    optim.zero_grad()
                    output = model(data)   

                    
                    targets.append(target)  
                    outputs.append(output)               
                    loss = self.loss(output, target).sum(1).mean(0)       
                    losses.update(loss.item()) 
                    loss.backward()

                    x=random.random() 
                    if x<prob and nums_visual>0:
                        nums_visual-=1
                        visual_ls.append( (data.detach().cpu().numpy(),output.detach().cpu().numpy() ))


                    optim.step()
                    if batch_idx % log_interval == 0:
                        err=compute_err(outputs,targets)
                        tepoch.set_postfix ( loss=loss.item(),err=err)
            error=compute_err(outputs,targets)
            print ("err train",error)

            # train_save="/root/repo/chexpert_classification/chexpert/output/learning_analysis/frontal_train.png"
            # train_note="/root/repo/chexpert_classification/chexpert/output/learning_analysis/frontal_train.txt"
            # plot_sample(train_save,train_note,visual_ls)

    def train(self):
        train_dataset=ChestDataset(cfg=self.cfg,mode="train",fold=self.fold)
    
        lowres_train_dataset=ChestDataset(cfg=self.cfg,mode="train", fold=self.fold,train_mode="progressive") if self.cfg["train_mode"]=="progressive" else None
        print(f'FOLD {self.fold}')
        print('--------------------------------')          
        train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=self.cfg["train_params"]["batch_size"])
        val_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.cfg["train_params"]["batch_size"])
        lowres_train_loader=torch.utils.data.DataLoader(
                    lowres_train_dataset, 
                    batch_size=self.cfg["train_params"]["batch_size"]) if lowres_train_dataset is not None else None
        

        train_metrics=[]
        val_metrics=[]
               
        if lowres_train_loader is not None :
            print(" TRAIN ON LOW-RES DATASET")     
            for epoch in range(1, self.cfg["epochs"] + 1):
                print(f"[]:  Epoch {epoch} of {self.cfg['epochs']} on Low-res dataset")
                self.train_epoch( lowres_train_loader,epoch,self.model,self.prog_optimizer)  
                self.prog_lr_scheduler.step()

        print(" TRAIN ON HI-RES DATASET")  
        if self.cfg ["train_params"]["early_stopping"]=="True":
            best_mean_AUC=- float('inf')
            epochs_no_improve = 0
            for epoch in range(1,self.cfg["frontal_epoch"]+1):
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
            for epoch in range(1,self.cfg["frontal_epoch"]+1):
                train_metric=self.train_epoch( train_loader,epoch,self.model,self.optim)                    
                self.lr_scheduler.step()
   
        print ("Finish default training")
        save_frontal_cls_path="/root/repo/chexpert_classification/chexpert/output/frontal_cls.pth"
        torch.save(self.model.state_dict(), save_frontal_cls_path)
        print('-'*100)

    
def compute_err(outputs,targets):

            out=torch.concat(outputs,dim=0).detach().view(-1)
            out=torch.sigmoid(out).float()
            out[out>0.5]=1.
            out[out<=0.5]=0.
            tar=torch.concat(targets,dim=0).detach().view(-1)
            out[out>0.5]=1.
            error=(abs(tar-out)).sum()/tar.shape[0]
            return error
      
def plot_sample (save_path,save_note,samples):
    assert (len(samples)>1) , "only plot for >1 samples"

    with open(save_note, 'w') as file:
         file.write("\n")

    fig,ax=plt.subplots(len(samples),figsize=(50,50))


    for i, sample in enumerate (samples):

        img=sample[0]
        pred=sample[1]

        # get a sample in a batch
        idx=random.choice(range(img.shape[0]))
        img=img[idx]
        pred=np.squeeze(pred[idx])
        pred[pred<0.5]=0
        pred[pred>0.5]=1
        pred_label="frontal" if pred ==1 else "not frontal"
        ax[i].imshow(np.transpose(img, [1,2,0]), label=pred_label)
        # ax[i].legend()
        with open(save_note,mode="a") as file:
            file.write( pred_label+"\n")
       
    plt.show()
    plt.savefig(save_path)
    plt.clf()
  
if __name__=="__main__":
    
     
    import yaml
    argconfig="/root/repo/chexpert_classification/chexpert/config/config.yaml"
    with open(argconfig, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    model_path="/root/repo/chexpert_classification/chexpert/output/frontal_cls.pth"
    net=chexpertNet(cfg=config,device="cuda",fold=1)
    
    net.train()
    state_dict = torch.load(model_path)


    net.model.load_state_dict(state_dict)
   
    set1=ChestDataset(cfg=config,mode="train")









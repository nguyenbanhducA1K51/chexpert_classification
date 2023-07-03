import torch.nn  as nn
import numpy as np
import torch
import sys
sys.path.append("../datasets")
sys.path.append("../model")
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from model import backbone
import matplotlib.pyplot as plt
import os
import glob
plt.style.use('ggplot')
from easydict import EasyDict as edict
import json
from datetime import datetime
from datasets.common import csv_index
import math 
class SaveBestModel:
    # this class only work for each training
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_AUC=-float('inf'),cfg
    ):
        self.best_valid_AUC = best_valid_AUC
        self.cfg=cfg
        
    def __call__(
        self, metric, 
        epoch, model, optimizer, criterion
    ):
        if int(cfg.mini_data)>=100000:

            abspath=os.path.dirname(os.path.abspath(__name__))+"/model/output/best_model.pth"
            if metric["meanAUC"] > self.best_valid_AUC:
                self.best_valid_AUC= metric["meanAUC"]
                print(f"\nBest validation  AUC: {self.best_valid_AUC}")
                print(f"\nSaving best model for epoch: {epoch}\n")
                torch.save({
                    'meanAUC':metric["meanAUC"],
                    'aucs': metric['aucs'],
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, abspath)
            

def save_plots(cfg,train_metric, val_metric):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # if int(cfg.mini_data)>=100000:
        classlabel=[]
        for x in cfg.class_idx:
            classlabel.append(csv_index[str(x)])
        now = datetime.now() 
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        parent_dir="/root/repo/Chexpert/chexpert/model/output/learning_analysis"
        folder=os.path.join(parent_dir,dt_string)
        os.mkdir(folder)
        train_classAUC=[]
        val_classAUC=[]

        train_meanAUC=[]
        val_meanAUC=[]

        train_loss=[]
        val_loss=[]
        for item in train_metric :
            train_classAUC.append(item["aucs"])
            train_meanAUC.append(item["meanAUC"])
            train_loss.append(item["loss"])
        for item in val_metric :
            val_classAUC.append(item["aucs"])
            val_meanAUC.append(item["meanAUC"])
            val_loss.append(item["loss"])

        train_classAUC=np.array(train_classAUC)
        val_classAUC=np.array(val_classAUC)

        train_meanAUC=np.array(train_meanAUC)
        val_meanAUC=np.array(val_meanAUC)

        train_loss=np.array(train_loss)
        val_loss=np.array(val_loss)
        row=int(math.sqrt(len(classlabel)+2))
        plt.figure(figsize=(20, 15))
        fig,ax=plt.subplots(nrows=row+1, ncols=row+1)
        stop_plot=False
        counter=0
        ax[0,0].plot(train_meanAUC,label="Mean train AUC")
        ax[0,0].plot(val_meanAUC,label="Mean validation AUC")
        ax[0,0].set(xlabel="epoch", ylabel="AUC")
        ax[0,0].set_title("Mean AUC")
        ax[0,1].plot(train_loss,label="Mean train loss")
        ax[0,1].plot(val_loss,label="Mean validation loss")
        ax[0,1].set(xlabel="epoch", ylabel="loss")
        ax[0,1].set_title("Mean loss")
        for i in range(row+1):
        if stop_plot:
            break
        for j in range (row+1):
            if i==0 and j==0:
                continue
            if i==0 and j==1:
                continue
            if stop_plot:
                break
            if counter>=len(classlabel):
                stop_plot=True
                break
            
            ax[i,j].plot(train_classAUC[:,counter],color='green', linestyle='-')
            ax[i,j].set(xlabel="epoch", ylabel="auc")
            ax[i,j].set_title(classlabel[counter])
            counter +=1

        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(path,"visualize.png"))

def calculateAUC (y_score,y_true,disease):
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

class Metric():
    def __init__(self,classes):
        self.classes=classes

    def compute_metrics(self,outputs, targets, losses=None):
        # shape work on tensor
        outputs=outputs.clone().cpu()
        targets=targets.clone().cpu()
        n_classes = outputs.shape[1]
        fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
        for i, clas in enumerate(self.classes):
            fpr[clas], tpr[clas], _ = roc_curve(targets[:,i], outputs[:,i])
            aucs[clas] = round (auc(fpr[clas], tpr[clas]) ,3)
            precision[clas], recall[clas], _ = precision_recall_curve(targets[:,i], outputs[:,i])
            fpr[clas], tpr[clas], precision[clas], recall[clas] = fpr[clas].tolist(), tpr[clas].tolist(), precision[clas].tolist(), recall[clas].tolist()       
        metrics = {
                    "meanAUC": round (np.mean(list(aucs.values())) ,4 ),
                    'fpr': fpr,
                'tpr': tpr,
                # 'aucs':  [np.round(item,2) for item in aucs],
                'aucs':aucs,
                'precision': precision,
                'recall': recall,
                'loss':losses
                
                }

        return metrics
class AverageMeter():
    def __init__(self,):
        self.reset()
        
    def reset(self):
        self.ls=[]
        self.mean=0
        self.cur=0
    def update (self, item):
        self.ls.append(item)
        self.mean= np.mean(self.ls)
        self.cur=item

def recordTraining(epoch=0,cfg=None, metric=None,transform=None):
    
    now = datetime.now()
    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    filePath=os.path.dirname(os.path.abspath(__name__))+"/model/output/recordTraining.csv"
    if int(cfg.mini_data.train)>=100000:
        with open(filePath, "a") as file:
        # Append some text to the file
            epochIndex=epoch
            meanAUC=metric["meanAUC"]
            listAUC= list(metric["aucs"].values())
            
            criterion=cfg.criterion
            if cfg.criterion=="balanceBCE":
                beta=cfg.balanceBCE.beta 
            else:
                beta=NA
            sample=cfg.mini_data.train
            totalEpoch=cfg.train.epochs
            op=cfg.train.optimizer.name
            lr=cfg.train.optimizer.lr
            if cfg.train_mode.name=="default":
                progressiveSample="NA"
                totalProgressiveEpoch="NA"
                progressiveOP="NA"
                progressivelr="NA"
            else:
                progressiveSample=cfg.progressive_mini_data.train
                totalProgressiveEpoch=cfg.progressive_train.epochs
                progressiveOP=cfg.progressive_train.optimizer.name
                progressivelr=cfg.progressive_train.optimizer.lr
            
            finalString=""
            finalString+=dt_string+","
            for auc in listAUC:
                finalString+=str(auc)+","

            finalString+=str(meanAUC)+","
            finalString+=str(cfg.backbone.name)+","
            finalString+=str(cfg.train_mode.name)+","
            finalString+=str(epochIndex)+","
            finalString+=str(sample)+","   
            finalString+=str(totalEpoch)+","
            
            finalString+=str(op)+","
            finalString +=str(lr)+","
            finalString+=str(criterion)+","
            finalString+=str(beta)+","

            finalString+=str(progressiveSample)+","   
            finalString+=str(totalProgressiveEpoch)+","
            
            finalString+=str(progressiveOP)+","
            finalString +=str( progressivelr)+","
            finalString +=str( cfg.image.progressive_image_size)+","
            finalString+=str(cfg.tta.usetta)+","
            finalString+=str(cfg.tta.times)

            # print (finalString)
            file.write('\n'+finalString)



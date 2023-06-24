import torch.nn  as nn
import numpy as np
import torch
import sys
sys.path.append("../datasets")
sys.path.append("../model")
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
import backbone
import matplotlib.pyplot as plt
import os
import glob
plt.style.use('ggplot')
from easydict import EasyDict as edict
import json

class SaveBestModel:
    # this class only work for each training
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_AUC=-float('inf')
    ):
        self.best_valid_AUC = best_valid_AUC
        
    def __call__(
        self, metric, 
        epoch, model, optimizer, criterion
    ):
        if metric["meanAUC"] > self.best_valid_AUC:
            self.best_valid_AUC= metric["meanAUC"]
            print(f"\nBest validation  AUC: {self.best_valid_AUC}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'meanAUC':metric["meanAUC"],
                'aucs': metric['aucs'],
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'output/best_model.pth')



    # model_state_dict = checkpoint['model_state_dict']
    # print (ckp)

def save_plots(train_aucs, valid_aucs, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_aucs, color='green', linestyle='-', 
        label='train AUC'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion AUC'
    )
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig('output/auc.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.png')

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

    def compute_metrics(self,outputs, targets, losses):
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
                    "meanAUC": np.mean(list(aucs.values())) ,
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


from datetime import datetime
def recordTraining(epoch=0,cfg=None, metric=None,transform=None):
    
    now = datetime.now()
    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    project_root=cfg.path.project_path
    filePath= project_root+"/model/output/recordTraining.txt"
  
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
        finalString +=str( progressivelr)
        print (finalString)
        file.write(finalString)


cfg_path="../config/config.json" 
# format: time,class1, ..classn, meanAUC, trainmode,numsample,epochindex,totalEpoch, criter,beta op,lr  
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(cfg_path) as f:
    cfg = edict(json.load(f))

metric={
    "meanAUC":0,
     "aucs" : {  "clasa":0.1,
    "clasb":0.2}
   }
recordTraining(metric=metric,cfg=cfg)
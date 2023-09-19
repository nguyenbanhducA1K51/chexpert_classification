import torch.nn  as nn
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import os
import glob
import json
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
import math 

sys.path.append("../")
from model import backbone
from data.common import csv_index



def save_metrics_and_models(metrics,models,fold):   

    train_metrics=metrics["train_stats"]
    val_metrics=metrics["val_stats"]
    


    folder=os.path.dirname(os.path.abspath(__name__))+"/output/"
    models_folder=os.path.join(folder,"models") 
    os.makedirs( models_folder ,exist_ok=True)
    plots_folder=os.path.join(folder,"plot") 
    os.makedirs( plots_folder ,exist_ok=True)
    # for i in range(len(models)):
    torch.save({
                'fold':fold,
                'train_metric':train_metrics,
                'val_metric':val_metrics,
                'mean_aucs_of_epochs':np.mean([data["meanAUC"] for data in val_metrics]),
                'highest_mean_auc':np.max([data["meanAUC"] for data in val_metrics] ),
                'model_state_dict': [model.state_dict()  for model in models],
                
                }, os.path.join(models_folder,f"fold_{fold}.pth"))
    save_plots(plots_folder,train_metrics,val_metrics,fold=fold)
               

def save_plots(folder,train_metrics, val_metrics,fold=1, class_idx=[ 7,10,11,13,15 ]):
            classlabel=[]
            for x in class_idx:
                classlabel.append(csv_index[str(x)])

            save_path=os.path.join(folder,f"metric_fold{fold}.png")
            
            train_meanAUC=[]
            val_meanAUC=[]
            train_loss=[]
            val_loss=[]

            train_aucs=[]
            val_aucs=[]

            for label in classlabel:
                
                train_label_list=[]
                val_label_list=[]
                for item in train_metrics :
                    train_label_list.append((item["aucs"][label]))
                for item in val_metrics :
                    val_label_list.append((item["aucs"][label]))
                train_aucs.append(train_label_list)
                val_aucs.append(val_label_list)
            train_aucs=np.array(train_aucs).reshape(-1,len(classlabel))
            val_aucs=np.array(val_aucs).reshape(-1,len(classlabel))

            for item in train_metrics :
                train_meanAUC.append(item["meanAUC"])
                train_loss.append(item["loss"])
            for item in val_metrics :
                val_meanAUC.append(item["meanAUC"])
                val_loss.append(item["loss"])

            train_meanAUC=np.array(train_meanAUC)
            val_meanAUC=np.array(val_meanAUC)

            train_loss=np.array(train_loss)
            val_loss=np.array(val_loss)
            row=int(math.sqrt(len(classlabel)+2))
            fig,ax=plt.subplots(row+1,row+1,figsize=(15, 15))
            stop_plot=False
            counter=0
            ax[0,0].plot(train_meanAUC,label="Mean train AUC")
            ax[0,0].plot(val_meanAUC,label="Mean validation AUC")
            ax[0,0].set(xlabel="epoch", ylabel="AUC")
            ax[0,0].set_title("Mean AUC")
            ax[0,0].legend()
            ax[0,1].plot(train_loss,label="Mean train loss")
            ax[0,1].plot(val_loss,label="Mean validation loss")
            ax[0,1].set(xlabel="epoch", ylabel="loss")
            ax[0,1].set_title("Mean loss")
            ax[0,1].legend()
            ax[2,1].remove()
            ax[2,2].remove()
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
                    ax[i,j].plot(train_aucs[:,counter],color='green', linestyle='-', marker="o",label="train auc")
                    ax[i,j].plot(val_aucs[:,counter],color='blue', linestyle='-', marker="o",label="val auc")
                    ax[i,j].set(xlabel="epoch", ylabel="auc")
                    ax[i,j].set_title(classlabel[counter])
                    ax[i,j].legend()
                    counter +=1
            fig.tight_layout()
            plt.show()
            fig.savefig(save_path)
    
def write_json(key,val,filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    data[key] = val
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)
        
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
                'aucs':aucs,
                'precision': precision,
                'recall': recall,
                'loss':losses
                
                }

        return metrics
    # def compute_frontal(self,outputs,targets):


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

class balanceBCE(nn.Module):
    def __init__(self,beta,device):
        super(balanceBCE,self).__init__()
        self.beta=beta
        self.device=device
    def forward(self,output,target):
        output=torch.sigmoid(output)
        ep=1e-10
        output=torch.clamp(output,min=ep, max=1-ep)
        positive=torch.sum(target,dim=0)
        negative= target.size()[0]-positive
        positive_factor= (1-self.beta)/ (1-self.beta**positive+1e-10)
        negative_factor=(1-self.beta)/ (1-self.beta**negative+1e-10)
        positive_factor=  positive_factor.unsqueeze(0).to(self.device)
        negative_factor= negative_factor.unsqueeze(0).to(self.device)
        positive_factor=torch.repeat_interleave(positive_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)
        negative_factor=torch.repeat_interleave(negative_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)
        loss=-positive_factor*target*torch.log(output)-(1-target)* negative_factor*torch.log(1-output)
        return loss




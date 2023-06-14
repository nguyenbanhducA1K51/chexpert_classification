
import sys
import torch
import json,os
sys.path.append("../datasets")
sys.path.append("../model")
import data
import modelUtils
import backbone
from torch.nn import functional as F
from easydict import EasyDict as edict
from PIL import Image

cfg_path="../config/config.json" 

with open(cfg_path) as f:
    cfg = edict(json.load(f))

num_class, train_loader,val_loader,test_loader=data.loadData(cfg=cfg,train_csv_path=cfg.path.train_csv_path,train_image_path=cfg.path.train_image_path,
 test_csv_path=cfg.path.test_csv_path,test_image_path=cfg.path.test_image_path,numPatient=cfg.numPatient, validation_split = cfg.validation_split,batch_size = cfg.train.batch_size
)
def train(model, criterion, optimizer, data_loader,  epoch,log_interval):
        train_correct=0
        counter=0
        train_loss=0
        
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            counter+=1
            optimizer.zero_grad()
            output = model(data)
            # pred = output.data.max(1, keepdim=True)[1]
            # train_correct += pred.eq(target.data.view_as(pred)).sum()
            train_correct+=compare(output=output,target=target)
            loss = criterion(output, target)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset), 
                    100. * batch_idx / len(data_loader),  loss.item()))
        print ("Train accuracy : {:.0f}% ".format (100. *train_correct/(len(data_loader.dataset)*num_class)))
        epoch_loss=  train_loss/counter
        epoch_acc=100. *train_correct/(len(data_loader.dataset)*num_class)
        return epoch_loss, epoch_acc
def compare(output,target):
    output=torch.sigmoid(output)
    output[output>=0.5]=1.
    output[output<0.5]=0
    return output.eq(target.view_as(output)).sum()
    

def eval(model,  data_loader , criterion):
     print ("VALIDATING :")
     model.eval()
     valid_running_loss = 0.0
     valid_running_correct = 0
     counter = 0
     
     with torch.no_grad():
        for data, target in data_loader:
            counter+=1
            output = model(data)
            loss = criterion(output, target)
            valid_running_loss += loss.item()
            valid_running_correct += compare(output=output,target=target)
    
     print('\nEval set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
     valid_running_correct, len(data_loader.dataset)*num_class,
    100. * valid_running_correct / (len(data_loader.dataset)*num_class) ))
     epoch_loss = valid_running_loss / counter
     epoch_acc = 100. * (valid_running_correct / (len(data_loader.dataset)*num_class ))
     return epoch_loss, epoch_acc

    
model,optimizer= modelUtils.loadModelAndOptimizer(numclass=num_class,cfg=cfg)
criterion=  F.binary_cross_entropy_with_logits
# self.best_valid_loss with infinity value when we 
# create an instance of the class. This is to ensure that any loss from the model will be less than the initial value.
save_best_model = modelUtils.SaveBestModel()

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(1, cfg.train.epochs + 1):
    print(f"[INFO]: Epoch {epoch} of {cfg.train.epochs}")
    train_epoch_loss, train_epoch_acc =  train(model=model, epoch=epoch, criterion=criterion, optimizer=optimizer, data_loader=train_loader, log_interval=cfg.train.log_interval)
    
    valid_epoch_loss, valid_epoch_acc =eval(model=model,data_loader=val_loader,criterion=criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    save_best_model(
        valid_epoch_loss, epoch, model, optimizer, criterion
    )
    print('-'*50)
# print (" train accu {} \n val accu {}".format(train_acc,valid_acc))
modelUtils.save_plots(train_acc, valid_acc, train_loss, valid_loss)







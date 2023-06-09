from math import exp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import os
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
sys.path.append("../datasets")
sys.path.append("../model")
import data
import utils
import backbone
from torch.nn import functional as F
     
load_dotenv()
# will put these param in arg parser
validation_split=0.2
batch_size=3
learning_rate = 0.01
momentum = 0.5
n_epochs = 2
log_interval=3

train_image_path= os.getenv ("TRAIN_IMAGE_PATH")
train_csv_path=os.getenv ("TRAIN_CSV_PATH")
test_image_path=os.getenv ("TEST_IMAGE_PATH")
test_csv_path=os.getenv ("TEST_CSV_PATH")
trans= {
   "train": transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor() ]),

        "val":transforms.Compose([
        transforms.Resize(256),
         transforms.ToTensor()
        
       
    ])
} 

num_class, train_loader,val_loader,test_loader=data.loadData(train_csv_path=train_csv_path,train_image_path=train_image_path,
 test_csv_path=test_csv_path,test_image_path=test_image_path,numPatient=50, transform=trans,validation_split = validation_split,batch_size = batch_size
)


def train(model, criterion, optimizer, data_loader,  epoch,log_interval=2):
        train_correct=0
        counter=0
        train_loss=0
        
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            counter+=1
            optimizer.zero_grad()
            output = model(data)
            pred = output.data.max(2, keepdim=True)[1]
            train_correct += pred.eq(target.data.view_as(pred)).sum()
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

def eval(model,  data_loader , criterion):
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
            pred = output.data.max(2, keepdim=True)[1]
            valid_running_correct += pred.eq(target.data.view_as(pred)).sum()
    
     print('\nEval set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
     valid_running_correct, len(data_loader.dataset)*num_class,
    100. * valid_running_correct / (len(data_loader.dataset)*num_class) ))
     epoch_loss = valid_running_loss / counter
     epoch_acc = 100. * (valid_running_correct / len(data_loader.dataset)*num_class)
     return epoch_loss, epoch_acc

    

model=backbone.VGGClassifier(num_class)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum) 
criterion=  utils.Loss()
save_best_model = utils.SaveBestModel()

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(1, n_epochs + 1):
    print(f"[INFO]: Epoch {epoch} of {n_epochs}")
    train_epoch_loss, train_epoch_acc =  train(model=model, epoch=epoch, criterion=criterion, optimizer=optimizer, data_loader=train_loader, log_interval=log_interval)
    
    valid_epoch_loss, valid_epoch_acc =eval(model=model,data_loader=val_loader,criterion=criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    save_best_model(
        valid_epoch_loss, epoch, model, optimizer, criterion
    )
    print('-'*50)
utils.save_model(n_epochs, model, optimizer, criterion)






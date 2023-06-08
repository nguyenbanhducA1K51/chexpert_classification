from math import exp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
sys.path.append("../datasets")
sys.path.append("../model")
import data
import utils
import backbone
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import random_split
     
load_dotenv()
# will put these param in arg parser
validation_split=0.2
batch_size=3
learning_rate = 0.01
momentum = 0.5
n_epochs = 1
log_interval=3

train_image_path= os.getenv ("TRAIN_IMAGE_PATH")
train_csv_path=os.getenv ("TRAIN_CSV_PATH")
test_image_path=os.getenv ("TEST_IMAGE_PATH")
test_csv_path=os.getenv ("TEST_CSV_PATH")
trans=transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),  ])
num_class, train_loader,val_loader,test_loader=data.loadData(train_csv_path=train_csv_path,train_image_path=train_image_path,
 test_csv_path=test_csv_path,test_image_path=test_image_path,numPatient=30, transform=trans,validation_split = validation_split,batch_size = batch_size
)

model=backbone.VGGClassifier(num_class)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum) 
criterion=  utils.Loss()

def train(model, criterion, optimizer, data_loader,  epoch,log_interval=2):
        correct=0
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            pred = output.data.max(2, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset), 
                    100. * batch_idx / len(data_loader),  loss.item()))
        print ("Train accuracy : {:.0f}% ".format (100. *correct/(len(data_loader.dataset)*num_class)))
def eval(model,  data_loader):
     model.eval()
   
     correct = 0
     with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
           
           
            pred = output.data.max(2, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    
     print('\nEval set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
     correct, len(data_loader.dataset)*num_class,
    100. * correct / (len(data_loader.dataset)*num_class) ))
    

def predict(model,sample):
    model.eval()
    with torch.no_grad():
        output=model(data)
       
        pred = output.data.max(2, keepdim=True)[1]
        return pred


for epoch in range(1, n_epochs + 1):

    train(model=model, epoch=epoch, criterion=criterion, optimizer=optimizer, data_loader=train_loader, log_interval=log_interval)
    
#     eval(model=model,data_loader=val_loader)
# eval(model=model,data_loader=test_loader)
count=10
while count >0:
    for _, (data, target) in enumerate(test_loader):
        model.eval()

        with torch.no_grad():
            output = model(data)


            pred = output.data.max(2, keepdim=True)[1]
            print ("output {} pred {}".format(output,pred))
        count-=1








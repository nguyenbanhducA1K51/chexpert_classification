from math import exp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
sys.path.append("../datasets")
import data
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
load_dotenv()

trainTransforms=transforms.Compose([
    
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  ])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   
valTransforms=transforms.Compose([
        transforms.Resize(320),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataset= data.ChestDataset(os.getenv("DATA_PATH")+"/train.csv",os.getenv("DATA_PATH")+"/train", transform=trainTransforms,numPatient=30)
batch_size = 3
validation_split = .2
shuffle_dataset = True
random_seed= 42
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
print ("len val loader {}".format(len(validation_loader)))
print ("len val dataset {}".format(len(validation_loader)))
print ("len train loader {}".format(len(train_loader)))
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, target):
        log_softmax=output-output.exp().sum(-1,keepdim=True).log()
        s1=list(output.size())[0]
        s2=list(output.size())[1]
        index1=np.array([[i]*s2 for i in range (s1)]).flatten()
        index2=[i for i in range(s2)]*s1
        index3= np.ravel(target)
        nll=-log_softmax[index1,index2,index3].mean()
        
        return nll
        




def predict(model,sample):
    model.eval()
    with torch.no_grad():
        output=model(data)
        prediction=output.numpy()
        prediction[prediction<0.5]=0
        prediction[prediction>=0.5]=1
        return prediction


class MultiDimLinear(torch.nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return out.reshape((len(x), *self.out_shape))

class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
       
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[6] = MultiDimLinear( in_features=4096, out_shape=(num_classes, 3))
        for name, param in self.vgg.named_parameters():
                if name!="classifier.6.weight" and name!="classifier.6.bias":
                    param.requires_grad = False
      
       
    
    def forward(self, x):
        x= self.vgg(x)
        return x
   
def train(model, criterion, optimizer, data_loader,  epoch,log_interval=2):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))
def eval(model, criterion, data_loader):
     model.eval()
     test_loss = 0
     correct = 0
     with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
           
            test_loss += criterion(output, target).item()
            pred = output.data.max(2, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
     test_loss /= len(data_loader.dataset)

     print('\nEval set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(data_loader.dataset)*data_loader.dataset.numclass,
    100. * correct / (len(data_loader.dataset)*data_loader.dataset.numclass ) ))



model=VGGClassifier(num_classes=14)
learning_rate = 0.01
momentum = 0.5
n_epochs = 3
log_interval=3
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)  
criterion=  Loss()

for epoch in range(1, n_epochs + 1):
  train(model=model, epoch=epoch, criterion=criterion, optimizer=optimizer, data_loader=train_loader, log_interval=2,)
  eval(model=model,criterion=criterion,data_loader=validation_loader)




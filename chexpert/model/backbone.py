import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
sys.path.append("../model")
import modelUtils

class DenseNetClassifier (nn.Module):
    def __init__ (self,num_classes):
        super(DenseNetClassifier, self).__init__()
        denseNet=models.densenet121(pretrained=True)
        self.dense=denseNet.features  
        self.pool=F.adaptive_avg_pool2d
        self.n_features= denseNet.classifier.in_features
        self.num_classes=num_classes
        
        self.generateClassificationLayer(in_feature=self.n_features)
    def forward (self,x):
        x=self.dense(x)
        x= F.relu(x, inplace=True)
        class_layer=getattr(self,"cls_0")
        logits=[]
        for i in range(0,self.num_classes):
            class_layer=getattr(self,"cls_"+str(i))
            z=self.pool(x,(1,1))
            z=torch.flatten(z,1)
            binary=class_layer(z)
            logits.append(binary)
        logits=torch.concat(logits,dim=1)    
        
        return logits
    def generateClassificationLayer(self,in_feature):
            for i in range (self.num_classes):
                setattr(self,"cls_"+str(i),nn.Linear(in_features=in_feature,out_features=1,bias=True))
    

class ConvNextClassifier (nn.Module):
    def __init__(self,num_classes):
        super(ConvNextClassifier, self).__init__()
        self.convnext=models.convnext_tiny()
        # weights='IMAGENET1K_V1'
        self.convnext.classifier[2]=nn.Linear(in_features=768, out_features=num_classes, bias=True)

       
    def forward(self,x):
        return self.convnext(x)






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
        # for name, param in self.dense.named_parameters():
        #         if  "classifier" not in name:
        #             param.requires_grad = False
        self.generateClassificationLayer(in_feature=self.n_features)
    def forward (self,x):
        x=self.dense(x)
        class_layer=getattr(self,"cls_0")
        z=self.pool(x,(1,1))
        # print ("z1 {}".format (z.size()))
        z=torch.flatten(z,1)
        logitTensor=class_layer(z)
        
        for i in range(1,self.num_classes):
            class_layer=getattr(self,"cls_"+str(i))
            z=self.pool(x,(1,1))
            # print ("z1 {}".format (z.size()))
            z=torch.flatten(z,1)
            binary=class_layer(z)
            logitTensor=torch.cat( (logitTensor,binary),dim=1)       
        return logitTensor
    def generateClassificationLayer(self,in_feature):
            for i in range (self.num_classes):
                setattr(self,"cls_"+str(i),nn.Linear(in_features=in_feature,out_features=1,bias=True))
       







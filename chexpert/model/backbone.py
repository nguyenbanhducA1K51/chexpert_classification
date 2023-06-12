import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import sys
sys.path.append("../model")
import modelUtils


class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # self.softmax=nn.Softmax(dim=2)
        # self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[6] = modelUtils.MultiDimLinear( in_features=4096, out_shape=(num_classes, 2))
        # for name, param in self.vgg.named_parameters():
        #         if  "classifier" not in name:
        #             param.requires_grad = False
         
    def forward(self, x):
        x= self.vgg(x)
        
        return x
class DenseNetClassifier (nn.Module):
    def __init__ (self,num_classes):
        super(DenseNetClassifier, self).__init__()
        self.dense=models.densenet121(pretrained=True)
        # self.dense.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features= self.dense.classifier.in_features
        self.dense.classifier=  modelUtils.MultiDimLinear( in_features=in_features, out_shape=(num_classes, 3)) 
        # self.softmax=nn.Softmax(dim=2)
        # for name, param in self.dense.named_parameters():
        #         if  "classifier" not in name:
        #             param.requires_grad = False

    def forward (self,x):
        x=self.dense(x)
        # x=self.softmax(x)
        return x
   






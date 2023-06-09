import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import utils


class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.softmax=nn.Softmax(dim=2)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[6] = utils.MultiDimLinear( in_features=4096, out_shape=(num_classes, 3))
        for name, param in self.vgg.named_parameters():
                if  "classifier" not in name:
                    param.requires_grad = False
         
    def forward(self, x):
        x= self.vgg(x)
        x=self.softmax(x)
        return x
# in progress
class DenseNetClassifier (nn.Module):
    def __init__ (self,num_classes):
        super(DenseNetClassifier, self).__init__()
        self.denseNet=models.densenet121(pretrained=True)
        self.softmax=nn.Softmax(dim=2)

    def forward (self,x):
        return x
   






from torchvision import models
import torch.nn as nn
import torch
class frontal_cls(nn.Module):
    def __init__(self):
        super(frontal_cls,self).__init__()
        self.backbone=models.resnet34(pretrained=True)
        self.backbone.fc=nn.Linear(512,1)
    def forward(self,x):
        return self.backbone (x)
    

    
if __name__=="__main__":
    # net=models.resnet34(pretrained=True)
    net=frontal_cls()
    # print (net)
    x=torch.rand(2,3,256,256)
    y=net(x)
    # print (y.shape)
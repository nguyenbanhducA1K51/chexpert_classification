import torch.nn  as nn
import numpy as np
import torch

class MultiDimLinear(torch.nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return out.reshape((len(x), *self.out_shape))

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


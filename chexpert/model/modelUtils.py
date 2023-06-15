import torch.nn  as nn
import numpy as np
import torch
import sys
sys.path.append("../datasets")
sys.path.append("../model")
from torch.nn import functional as F
# Convnext

# import data
# import utils
import backbone
import matplotlib.pyplot as plt
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
        # shape output : N*M*2 where N is batch size, M is number of disease, which is 14 in this dataset
        log_softmax=output-output.exp().sum(-1,keepdim=True).log()

        s1=list(output.size())[0] # N

        s2=list(output.size())[1] # M
        index1=np.array([[i]*s2 for i in range (s1)]).flatten()
        # for example, if s1=7, s2=3 then 
        #index1 =[0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6]
        #index2= [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        index2=[i for i in range(s2)]*s1
        #
        index3= np.ravel(target)
        nll=-log_softmax[index1,index2,index3].mean()
    
        
        return nll   
        


plt.style.use('ggplot')
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'output/best_model.pth')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'output/final_model.pth')

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('output/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.png')

def loadModel (cfg,num_class):
    if cfg.backbone=="densenet121":
        model=backbone.DenseNetClassifier(numclass)
    elif cfg.backbone=="vgg":
        model =backbone.VGGClassifier(numclass)
    if cfg.optimizer.name=="Adam":
        op= torch.optim.Adam(model.parameters(),lr=cfg.optimizer.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif cfg.optimizer.name=="SGD":
        op=torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum) 
    return model,op
#     



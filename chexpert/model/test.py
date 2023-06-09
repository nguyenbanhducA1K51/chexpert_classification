import torch
from tqdm.auto import tqdm
from model import build_model
from datasets import create_datasets, create_data_loaders


def test(model, testloader,numclass):
    """
    Function to test the model
    """
    # set model to evaluation mode
    model.eval()
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for _, (image,target) in (enumerate(testloader)):
            counter += 1
            
        
            output = model(image)
          
            pred = output.data.max(2, keepdim=True)[1]
            valid_running_correct += (pred == target).sum().item()
        
    # loss and accuracy for the complete epoch
    final_acc = 100. * (valid_running_correct / ( len(testloader.dataset))*numclass )
    return final_acc

# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader)
    print(f"Best epoch saved model accuracy: {test_acc:.3f}")
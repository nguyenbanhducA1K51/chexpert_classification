import os
import torch
import copy
def test(self):
        print(f"EVALUATING ON TEST SET")
        test_dataset= ChestDataset(cfg=self.cfg,mode="test") 
        data_loader=torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size=self.cfg.train.batch_size)
        current_file_path = os.path.abspath(__file__)

        parent_directory = os.path.dirname(current_file_path)
        models_save_path=os.path.join (parent_directory, "output/models")
        
        files = [f for f in os.listdir(models_save_path) if os.path.isfile(os.path.join(models_save_path, f))]
        pth_files = [f for f in files if f.endswith('.pth')]
        models=[]
        for file in pth_files:
            path=os.path.join(models_save_path,file)
            state_dict=torch.load(path)
            new_model=copy.deepcopy(self.model)
            new_model.load_state_dict(state_dict['model_state_dict'])
            models.append(new_model)
        for model in models:
            model.to(self.device)
            model.eval()
        losses= utils.AverageMeter()   
        outputs=[]
        targets=[]
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for idx,(data, target) in enumerate(tepoch):
                    data=data.to(self.device).float()
                    target=target.to(self.device)              
                    targets.append(target)
                    tmp_output =[model(data) for model in models]
                    output=torch.mean(torch.stack(tmp_output), dim=0)
                    outputs.append(output)        
                    loss = self.criterion(output, target).sum(1).mean(0)    
                    losses.update(loss.item()) 
                    if idx%4==0:
                        tepoch.set_postfix(loss=loss.item())                                     
        outputs=torch.concat(outputs,dim=0).detach()
        targets=torch.concat(targets,dim=0).detach()
        metric=self.metric.compute_metrics(outputs=outputs,targets=targets,losses=losses.mean)
        print (" Mean AUC : {: .3f}. AUC for each class: {}".format(metric["meanAUC"],metric["aucs"]))  

from pathlib import  Path
import sys
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import os
def is_frontal(x):
        return 1 if "frontal" in x else 0 
    
if __name__=="__main__":

    config_path= Path(__file__).resolve().parent /"config/config.yaml"
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    root=config["path"]["data_path"]

    train_path=os.path.join(root,"CheXpert-v1.0-small/train.csv")
    process_train= config["path"]["process_train"]
    valid_path= os.path.join(root,"CheXpert-v1.0-small/valid.csv")
    process_valid=config["path"]["process_test"]

   
    df=pd.read_csv(train_path)
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids,val_ids) in enumerate(kfold.split(range(len(df))) ):
        df.loc[val_ids,"fold"]=str(fold+1)
    

    df["is_frontal"]=df["Path"].apply(is_frontal)#
    df.to_csv(process_train,index=False)
    print ("Finish split k_fold")

  
    df=pd.read_csv(valid_path)
    df["is_frontal"]=df["Path"].apply(is_frontal)
    df.to_csv(process_valid,index=False)
    







    
    



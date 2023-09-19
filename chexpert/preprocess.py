from pathlib import  Path
import sys
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import os

if __name__=="__main__":
    config_path= Path(__file__).resolve().parent /"config/config.yaml"
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    train_csv=config["path"]["train_csv_path"]
    save_k_fold=config["path"]["save_k_fold"]
    # if os.path.exists(save_k_fold):
    #      os.remove(save_k_fold)
    # df=pd.read_csv(train_csv)
    # kfold = KFold(n_splits=5, shuffle=True)
    # for fold, (train_ids,val_ids) in enumerate(kfold.split(range(len(df))) ):
    #     df.loc[val_ids,"fold"]=fold+1
    # df.to_csv(save_k_fold,index=False)
    # print ("Finish split k_fold")
    def is_frontal(x):
        return 1 if "frontal" in x else 0

# Apply the function to generate a new column, say Column2
    # df['Column2'] = df['Column1'].apply(my_function)
    df=pd.read_csv(save_k_fold)
    print (df.head().Path.to_list())
    df["is_frontal"]=df["Path"].apply(is_frontal)
    # # print (df.loc[:, ["Path", "is_frontal"]].head(10))
    # print (df["Path"].head(10).to_list())
    # print (df["is_frontal"].head(10).to_list())
    df.to_csv(save_k_fold,index=False)


    
    



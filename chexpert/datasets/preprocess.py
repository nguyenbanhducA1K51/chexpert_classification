import os
from dotenv import load_dotenv
import pandas as pd 
import glob as glob
import numpy as np
from PIL import Image
import cv2
import random
load_dotenv()
def f1(numTrainAndVal, percentval):
    train_num= numTrainAndVal*(1-percentval)
    
    root_dir=os.getenv("DATA_PATH")
    subdir= sorted(os.listdir(root_dir+"/train"))[1:numTrainAndVal+1]

    train_path=root_dir+"/train"
    img_path=[]
    for dir in subdir:
            img_path+= glob.glob("{}/{}/*/*.jpg".format(train_path,dir)) 
    label_path= root_dir+"/train.csv"
    labels=pd.read_csv(label_path,nrows=len(img_path)+1)
    labels=np.array(labels.iloc[1:,5:])
    tuple_list=[]
    for i in range (labels.shape[0]):
        tuple_list.append(img_path[i],labels[i])
    return tuple_list[:train_num] , tuple_list[train_num:]




def image_trainval(numTrainAndVal, percentval):
    train_num= numTrainAndVal*(1-percentval)
    
    root_dir=os.getenv("DATA_PATH")
    subdir= sorted(os.listdir(root_dir+"/train"))[1:numTrainAndVal+1]

    train_path=root_dir+"/train"
    img_path=[]
    for dir in subdir:
            img_path+= glob.glob("{}/{}/*/*.jpg".format(train_path,dir)) 
            
    img_list=np.zeros((len(img_path),390,320,3))

    for i in range (len(img_path)):
        img=cv2.imread(img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized=cv2.resize(img, (320,390), interpolation=cv2.INTER_NEAREST)
        img_list[i]=resized

    label_path= root_dir+"/train.csv"
    labels=pd.read_csv(label_path,nrows=len(img_path)+1)
    labels=np.array(labels.iloc[1:,5:])

    print (img_list[0])
    tuples_list=[]
    for i in range (labels.shape[0]):
        tuples_list.append((img_list[i],labels[i]))
    random.shuffle(tuples_list)
    return (tuples_list[0:train_num],tuples_list[train_num:])

def image_test(size):
    root_dir=os.getenv("DATA_PATH")
    subdir= sorted(os.listdir(root_dir+"/"))[1:size+1]

    valid_path=root_dir+"/valid"
    img_path=[]
    for dir in subdir[0:3]:
            img_path+= glob.glob("{}/{}/*/*.jpg".format(valid_path,dir)) 
            
    img_list=np.zeros((len(img_path),390,320,3))

    for i in range (len(img_path)):
        img=cv2.imread(img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized=cv2.resize(img, (320,390), interpolation=cv2.INTER_NEAREST)
        img_list[i]=resized

    label_path= root_dir+"/train.csv"
    labels=pd.read_csv(label_path,nrows=len(img_path)+1)
    labels=np.array(labels.iloc[1:,5:])

    print (img_list[0])
    tuples_list=[]
    for i in range (labels.shape[0]):
        tuples_list.append((img_list[i],labels[i]))
    return tuples_list


## exper
root_dir=os.getenv("DATA_PATH")
# subdir= sorted(os.listdir(root_dir+"/train"))[0:6]
subdir= sorted(os.listdir(root_dir+"/valid"))[0:6]
img_path=[]
for dir in subdir[0:3]:
            img_path+= glob.glob("{}/valid/{}/*/*.jpg".format(root_dir,dir)) 
print (img_path)

image = Image.open(img_path[0]) 







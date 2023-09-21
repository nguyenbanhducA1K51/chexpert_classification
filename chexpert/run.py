
import sys
import torch
import json,os
from src import chexpert
from torch.nn import functional as F
from easydict import EasyDict as edict
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--fold', type=str, default=1)
    parser.add_argument('--mode',type=str,default="train")
    return parser.parse_args()

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    net=chexpert.chexpertNet(cfg=config,device=device,fold=args.fold)

    net.train()
  











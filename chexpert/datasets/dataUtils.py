import cv2
import numpy as  np
import torchvision.transforms as transform
from easydict import EasyDict as edict
import json,os


def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.image_fix_length
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.image_fix_length
        h_ = round(w_ / ratio)
    # print(" h {} w{}".format(h_,w_ ))
    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image=np.pad(image, ((0, cfg.image_fix_length - image.shape[0]),
                               (0, cfg.image_fix_length - image.shape[1]), (0, 0)),
                       mode='constant',
                       constant_values=cfg.pixel_mean)
    # print(image.shape)
    return image

def defaultTransform(image, cfg):
    # Transformimage in form of numpy array
    assert image.ndim == 2, "image must be gray image"

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, cfg)
    # normalization
    image = image.astype(np.float32) - cfg.pixel_mean
    if cfg.pixel_std:
        image /= cfg.pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image
def trainTransform(image):
    aug=transform.Compose([
        # transform.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
        #                  scale=(0.95, 1.05), fillcolor=128)
        transform.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05))
    ])
    image=aug(image)
    return image

im=np.random.rand(500,420,3)
cfg_path="../config/config.json" 

with open(cfg_path) as f:
    cfg = edict(json.load(f))
fix_ratio(im,cfg)

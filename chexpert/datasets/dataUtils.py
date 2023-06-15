import cv2
import numpy as  np
import torchvision.transforms as transform
from easydict import EasyDict as edict
import json,os
from PIL import Image
import matplotlib.pyplot as plt


def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.image.image_fix_length
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.image.image_fix_length
        h_ = round(w_ / ratio)
    # print(" h {} w{}".format(h_,w_ ))
    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image=np.pad(image, ((0, cfg.image.image_fix_length - image.shape[0]),
                               (0, cfg.image.image_fix_length - image.shape[1]), (0, 0)),
                       mode='constant',
                       constant_values=cfg.image.pixel_mean)
    # print(image.shape)
    return image

def defaultTransform(image, cfg):
    # Transformimage in form of numpy array
    assert image.ndim == 2, "image must be gray image"

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, cfg)
    # normalization
    image = image.astype(np.float32) - cfg.image.pixel_mean
    if cfg.image.pixel_std:
        image /= cfg.image.pixel_std
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
        # gaussian noise,.etc
        # 
    ])
    image=aug(image)
    return image





# cfg_path="./config/config.json" 

# with open(cfg_path) as f:
#     cfg = edict(json.load(f))
# path="/Users/mac/Downloads/CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"

# image=cv2.imread(path,0)

# image = Image.fromarray(image)
# # image.show()
# image=trainTransform(image=image)
# image.show()
# # print (image.shape)
# image=np.array(image)
# print (image.shape)
# image=defaultTransform(image,cfg=cfg)

# print (image.shape)
# image=image.transpose( (1,2,0))
# plt.imshow(image)
# plt.show()
# image = Image.fromarray(image)
# image.show()

# to show an image, first use cv2.imread(path,0), it will return an numpy array
# then use Image.fromarray(image), which will convert to PIL image,then can show it use
#image.show()

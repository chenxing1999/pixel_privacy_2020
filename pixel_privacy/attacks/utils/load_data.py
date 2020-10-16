from torchvision import models, transforms
import torch
from torch import nn
import torchvision.transforms.functional as Ftransform


import warnings
from PIL import Image


from flashtorch.utils import format_for_plotting, standardize_and_clip
from matplotlib import pyplot as plt
from torch.nn import functional as F

import os
import json



IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD  = [0.229, 0.224, 0.225]

#IMAGENET_MEAN.reverse()
#IMAGENET_STD.reverse()

def load_img(fname):
  ''' Load image + normalize image to tensor
  Input:
    fname: (str) Image file path
  
  Output:
    - Tensor (1, 3, H, W)
  
  
  '''
  
  img_pil = Image.open(fname)
  img_pil = img_pil.convert('RGB')
  transform = transforms.Compose(
      [
      #Havent add normalize function due to unknown mean and std
      
      transforms.Resize(256),
      transforms.ToTensor(),
      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
      ]
  
  
  )
  img_tensor = transform(img_pil)
  #img_tensor = img_tensor[(2, 1, 0), :, :]
  img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
  return img_tensor
  
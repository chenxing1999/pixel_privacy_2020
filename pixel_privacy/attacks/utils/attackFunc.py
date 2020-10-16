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

from .load_data import load_img, IMAGENET_MEAN, IMAGENET_STD




invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                    std = [ 1., 1., 1. ]),
                              ])

def do_inverse_IMNET(tensor):
  return invTrans(tensor.squeeze(0))[None]


def process_single_image_1(fname, model, target_class=None, lr=9e-4, output_path=None, ilen=5, jlen=201):
  img = load_img(fname)
  img_mean = img.mean()
  img_std = img.std()

  tensor = img.cuda()
  model.cuda()
  prev_tensor = tensor
  model.eval()
  optimizer = None
  flag = False

  ## Load image one time
  # print(torch.argmax(model(tensor)))
  if target_class is None:
    target_class = int(torch.argmax(model(tensor)))

  for i in range(ilen):
    with torch.no_grad():
      if not optimizer is None:
        tensor = optimizer.param_groups[0]['params'][0]
      tensor = tensor.sub(tensor.mean()).div(tensor.std()).mul(img_std).add(img_mean)
      # tensor = tensor.clamp(0, 1)
    tensor.requires_grad_(True)
    optimizer = torch.optim.Adam([tensor], lr)
    
    for j in range(jlen):
      model_output = model(tensor)
      target = torch.zeros_like(model_output)
      target[0, target_class] = 1
      
      optimizer.zero_grad()
      model_output.backward(target)
      optimizer.step()
      
      if j == 0:
        print(F.softmax(model_output[0])[target_class])
      
      if torch.argmax(model_output[0]) != target_class:
        print(j)
        print(F.softmax(model_output[0])[target_class])
        flag = True
        break

    if flag:
      break
  with torch.no_grad():
    if not optimizer is None:
      tensor = optimizer.param_groups[0]['params'][0]
    tensor = tensor.sub(tensor.mean()).div(tensor.std()).mul(img_std).add(img_mean)
    # tensor = tensor.clamp(0, 1)
  print(tensor.max())
  tensor = do_inverse_IMNET(tensor)
  plt.imshow(format_for_plotting(tensor.cpu()[0]))

# Thang's version
from IPython.display import display

def process_single_image_2(fname, model, target_class=None, lr=9e-4, output_path=None, ilen=5, jlen=201,do_show=True):
  transform = transforms.Compose(
      [
      #Havent add normalize function due to unknown mean and std
      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
      ]
  )
  
  
  img = load_img(fname)
  img_mean = img.mean()
  img_std = img.std()
  
  tensor = img.cuda()
  model.cuda()
  prev_tensor = tensor
  model.eval()
  optimizer = None
  flag = False
  tensor1 = do_inverse_IMNET(tensor)
  # print(tensor.max(),tensor.min(),tensor.mean(),tensor.std())
  # print(tensor1.max(),tensor1.min(),tensor1.mean(),tensor1.std())

  ## Load image one time
  # print(torch.argmax(model(tensor)))
  if target_class is None:
    target_class = int(torch.argmax(model(tensor)))

  for i in range(ilen):
    # with torch.no_grad():
    #   if not optimizer is None:
    #     tensor = optimizer.param_groups[0]['params'][0]
    #   tensor = tensor.sub(tensor.mean()).div(tensor.std()).mul(img_std).add(img_mean)
    #   tensor = tensor.clamp(-2.64, 2.64)
    # tensor.requires_grad_(True)
    # optimizer = torch.optim.Adam([tensor], lr)
    
    for j in range(jlen):
      with torch.no_grad():
        if not optimizer is None:
          tensor = optimizer.param_groups[0]['params'][0]
        
        tensor = tensor.sub(tensor.mean()).div(tensor.std()).mul(img_std).add(img_mean)
        tensor = tensor.clamp(-2.1179, 2.64)
        tensor = do_inverse_IMNET(tensor)
        tensor = tensor.clamp(0, 1)
        tensor = transform(tensor.squeeze(0))[None]

      tensor.requires_grad_(True)
      optimizer = torch.optim.Adam([tensor], lr)
      model_output = model(tensor)

            


      target = torch.zeros_like(model_output)
      target[0, target_class] = 1
      optimizer.zero_grad()
      model_output.backward(target)
      optimizer.step()
      
      if torch.argmax(model_output[0]) != target_class:
        # if torch.argmax(model(tensor)) != target_class:
        print(tensor.max(),tensor.min(),tensor.mean(),tensor.std())
        break

      if j == 0:
        print('score hien tai ', i, '-',F.softmax(model_output[0])[target_class])


    if flag:
      break

  with torch.no_grad():
    # if not optimizer is None:
    #   tensor = optimizer.param_groups[0]['params'][0]
    
    tensor = tensor.sub(tensor.mean()).div(tensor.std()).mul(img_std).add(img_mean)
    tensor = do_inverse_IMNET(tensor)
    tensor = tensor.clamp(0, 1)
  # print(tensor.max(),tensor.min(),tensor.mean(),tensor.std())
  print(tensor.max(),tensor.min(),tensor.mean(),tensor.std())
  if do_show:
    # plt.imshow(format_for_plotting(tensor.cpu()[0]))
    showim = Ftransform.to_pil_image(tensor.cpu()[0])
    display(showim)
  if output_path is not None:

    saveim = Ftransform.to_pil_image(tensor.cpu()[0]).convert('RGB')
    saveim.save(output_path, 'PNG',compress_level=0)
    tensor = transform(tensor.squeeze(0))[None]
    print('before save',torch.argmax(model(tensor)))
    tmp_im = load_img(output_path).cuda()
    print('after save ',torch.argmax(model(tmp_im)))
    print('max delta ', (abs(tensor-tmp_im)).max())
# -*- coding: utf-8 -*-
"""NSTmain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c6qn9ml8FiLonAPMIeTx4k4_i16OfBZN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms
import torch.optim as optim
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

"""LOADING MODEL"""

vgg=models.vgg19(pretrained=True).features

for parameter in vgg.parameters():
    parameter.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

"""IMAGE LAODER AND TRANSFROMS"""

from google.colab import drive
drive.mount('/content/drive')

style_img_path1="/content/drive/MyDrive/Images/Starry-Night-canvas-Vincent-van-Gogh-New-1889.jpeg"
style_img_path2="/content/drive/MyDrive/Images/COLORFUL-NIGHT.jpg"
content_img_path="/content/drive/MyDrive/Images/Tuebingen_Neckarfront.jpg"

content_img=Image.open(content_img_path).convert('RGB')
style_img1=Image.open(style_img_path1).convert('RGB')
style_img2=Image.open(style_img_path2).convert('RGB')

size=450

def imageLoader(img,size):
  img_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

  img=img_transform(img).unsqueeze(0)

  return img


def imageUnLoader(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

imgs_tensor=[imageLoader(content_img,size).to(device),imageLoader(style_img1,size).to(device),imageLoader(style_img2,size).to(device)]
content,style1,style2=imgs_tensor

# display the images
fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
# content and style ims side-by-side
ax1.imshow(imageUnLoader(content))
ax2.imshow(imageUnLoader(style1))
ax3.imshow(imageUnLoader(style2))

for name, layer in vgg._modules.items():
  print(name)
  print(layer)

def featureMapExtractor(image,model,layers):
  if layers=="style":
    layers=["1", "6", "11", "20", "29"]

  if layers=="content":
    layers= ["22"]

  if layers=="generated":
    layers=["22","1", "6", "11", "20", "29"]


  features={}
  x=image
  for name, layer in model._modules.items():
    if isinstance(layer, nn.ReLU):
        x = F.relu(x, inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        x = nn.AvgPool2d(2, 2)(x)
    else:
        x = layer(x)

    if name in layers:
        features[name] = x
  return features

"""Defining GramMatrix"""

def gramMatrix(tensor):
  b,d,h,w=tensor.size()

  tensor=tensor.view(b*d,h*w)

  gram = torch.mm(tensor, tensor.t())

  return gram

content_fms=featureMapExtractor(content,vgg,"content")
style_fms1=featureMapExtractor(style1,vgg,"style")
style_gram1={layer: gramMatrix(style_fms1[layer]) for layer in style_fms1}
style_fms2=featureMapExtractor(style2,vgg,"style")
style_gram2={layer: gramMatrix(style_fms2[layer]) for layer in style_fms2}
generated_img=content.clone().requires_grad_(True).to(device)

style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
alpha = 1e0
beta=1e6
gamma=0.25
style_layers=["1", "6", "11", "20", "29"]

"""Style Transfer"""

num_of_steps=8000
show_iter=1000
step=0

optimizer = optim.Adam(params=[generated_img], lr=0.003)



while step <= num_of_steps:

    generated_img_features = featureMapExtractor(generated_img, vgg, "generated")

    content_loss = torch.mean((generated_img_features["22"] - content_fms["22"])**2)

    style_loss1 = 0
    i = 0
    for layer in style_layers:
        generated_img_feature = generated_img_features[layer]
        generated_img_gram = gramMatrix(generated_img_feature)
        _, d, h, w = generated_img_feature.shape
        style_img_gram1 = style_gram1[layer]
        layer_style_loss = style_weights[i] * torch.mean((generated_img_gram - style_img_gram1)**2)
        style_loss1 += layer_style_loss / (d * h * w)
        i += 1

    style_loss2 = 0
    i = 0
    for layer in style_layers:
        generated_img_feature = generated_img_features[layer]
        generated_img_gram = gramMatrix(generated_img_feature)
        _, d, h, w = generated_img_feature.shape
        style_img_gram2 = style_gram2[layer]
        layer_style_loss = style_weights[i] * torch.mean((generated_img_gram - style_img_gram2)**2)
        style_loss2 += layer_style_loss / (d * h * w)
        i += 1

    total_loss = alpha * content_loss + beta * (gamma*style_loss1+(1-gamma)*style_loss2)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()



    if step % show_iter == 0:
        print('Step {}: Total loss: {:.4f}'.format(step, total_loss.item()))
        plt.imshow(imageUnLoader(generated_img))
        plt.show()

    step += 1
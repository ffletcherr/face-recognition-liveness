"""
Created on Sat Nov  7 00:49:37 2020

@author: homayoun
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
from glob import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

resnet = InceptionResnetV1(pretrained='vggface2').eval()
pil2tensor = transforms.ToTensor()
plt.ioff()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
images_list = sorted(glob('hm/*'))
faces  =  []
names = []
for j,image_path in enumerate(images_list): 
    
    names.append(image_path.split('/')[-2]+'-{}'.format(image_path.split('/')[-1].split('.')[0]))
    img = Image.open(image_path).convert('RGB')
    img_cropped = pil2tensor(img)
    faces.append(img_cropped)
print('bank created')   
aligned = torch.stack(faces).to(device)
embeddings = resnet(aligned).detach().cpu().numpy()
facebank = pd.DataFrame(embeddings,  index=names)
facebank.to_csv('hm.csv',header=None,index=None)










"""
Created on Sat Nov  7 02:22:16 2020

@author: homayoun
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
plt.ioff()
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
i = 0
th1 = .7
th2 = .9
input_folder= 'images'
output_folder = 'lfw-check'
facebank = pd.read_csv('hm.csv',header=None)
images_list = glob(os.path.join(input_folder,'*.png')) + glob(os.path.join(input_folder,'*.jpg'))
print('begin real-time process')
for image_path in tqdm(images_list):
    image_name = image_path.split('/')[-1].split('.')[0]
    verify = False
    img = Image.open(image_path).convert('RGB')
    i += 1
    # detection
    try:
        img_cropped, boxes = mtcnn(img)
    except :
        continue
    embeddings = resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()
    diff = facebank - embeddings
    norm = np.linalg.norm(diff,axis=1)
    norm_min = np.around(norm.min(),3)
    norm_mean = np.around(norm.mean(),3)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.title(f'min:{norm_min}, mean:{norm_mean}')
    if norm.min() < th1 and norm.mean() < th2 :
        plt.savefig(os.path.join(output_folder,'VERIFIED_'+image_name))
    else:
        plt.savefig(os.path.join(output_folder,image_name))
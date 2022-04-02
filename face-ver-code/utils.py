"""
Created on Wed Nov 11 20:15:26 2020

@author: homayoun
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from time import time 
from facenet_pytorch import MTCNN, InceptionResnetV1

class faceMTCNN():

    def __init__(self,period=5):
        self.period = period
        self.time   = time()
        self.buff   = False
        self.mtcnn  = MTCNN()
        
        
    
    def schedule(self,img):
        if time() - self.time > self.period:
            self.buff = True
            crop, boxes = self.mtcnn(img)
            self.time   = time()
            return crop, boxes
        else:
            self.buff = False
            return None,None
        
class embedResnet():
    def __init__(self,period=15):
        self.period = period
        self.time   = time()
        self.last_face = None
        self.resnet  = InceptionResnetV1(pretrained='vggface2').eval()
        
    def schedule(self):
        if time() - self.time > self.period:
            crop = self.last_face
            if crop is not None:
                embeddings = self.resnet(crop.unsqueeze(0)).detach().cpu().numpy()
                self.time   = time()
                return embeddings
            else:
                return None
        else:
            return None
        

import os
import statistics as st
import sys
import time
import traceback
from copy import deepcopy
from glob import glob
from threading import Thread
import onnxruntime
import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from torchvision import transforms

from asset import detect_faces_mediapipe, extract_face, post_process

plt.ioff()
th1 = .75
th2 = .95
detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, static_image_mode=True)
resnet_onnx = onnxruntime.InferenceSession(
    "InceptionResnetV1_vggface2.onnx", providers=['CPUExecutionProvider'])

facebank = pd.read_csv('hm.csv', header=None)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bboxs = detect_faces_mediapipe(detector, frame)
    if not len(bboxs):
        continue
    bbox = bboxs[0]
    face_arr = extract_face(frame, bbox.flatten())
    face_tensor = post_process(face_arr)
    embeddings = resnet_onnx.run(["output"],
                                 {"input": face_tensor.unsqueeze(0).detach().cpu().numpy().astype(np.float32)})[0]
    diff = facebank - embeddings
    norm = np.linalg.norm(diff, axis=1)
    norm_min = np.around(norm.min(), 3)
    norm_mean = np.around(norm.mean(), 3)
    if norm.min() < th1 and norm.mean() < th2:
        verify = True
    else:
        verify = False
    print("verify:", verify)
    cv2.imshow("face", face_arr)
    k = cv2.waitKey(2)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()


import os
import statistics as st
import sys
import time
import traceback
from copy import deepcopy
from glob import glob
from threading import Thread

import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import onnxruntime
import pandas as pd
import psutil
from PIL import Image
from torchvision import transforms

from asset import detect_faces_mediapipe, extract_face, post_process, trans

plt.ioff()
th1 = .75
th2 = .95
detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, static_image_mode=True)
deepPix_onnx = onnxruntime.InferenceSession(
    "../Pretrained_models/OULU_Protocol_2_model_0_0.onnx", providers=['CPUExecutionProvider'])

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
    face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = trans(face_pil)
    output_pixel, output_binary = deepPix_onnx.run(["output_pixel", "output_binary"],
                                                   {"input": face_tensor.unsqueeze(0).detach().cpu().numpy().astype(np.float32)})
    liveness_score = (np.mean(output_pixel.flatten()) +
                      np.mean(output_binary.flatten()))/2.0
    print("liveness_score:", liveness_score)
    cv2.imshow("face", face_arr)
    k = cv2.waitKey(2)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

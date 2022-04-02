import os
from typing import List, Tuple
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime
import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchvision import transforms as T


class FaceDetection():
    def __init__(self):
        self.detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, static_image_mode=True)

    def __call__(self, image) -> Tuple[List[np.ndarray], List[List[int]]]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        boxes = []
        faces = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                face_arr = extract_face(image, bbox.flatten().tolist())
                boxes.append(bbox)
                faces.append(face_arr)
        return faces, boxes


class IdentityVerification():
    def __init__(self, checkpoint_path: str, facebank_path: str):
        self.resnet = onnxruntime.InferenceSession(
            checkpoint_path, providers=['CPUExecutionProvider'])
        self.facebank = pd.read_csv(facebank_path, header=None)

    def __call__(self, face_arr: np.ndarray) -> Tuple[float, float]:
        face_arr = np.moveaxis(face_arr, -1, 0)
        input_arr = np.expand_dims((face_arr - 127.5) / 128.0, 0)
        embeddings = self.resnet.run(["output"],
                                     {"input": input_arr.astype(np.float32)})[0]
        diff = self.facebank - embeddings
        norm = np.linalg.norm(diff, axis=1)
        min_sim_score = np.around(norm.min(), 3)
        mean_sim_score = np.around(norm.mean(), 3)
        return min_sim_score, mean_sim_score


class LivenessDetection():
    def __init__(self, checkpoint_path: str):
        self.deepPix = onnxruntime.InferenceSession(
            checkpoint_path, providers=['CPUExecutionProvider'])
        self.trans = T.Compose([T.Resize((224, 224)),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]
                               )

    def __call__(self, face_arr: np.ndarray) -> float:
        face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.trans(face_pil).unsqueeze(0).detach().cpu().numpy()
        output_pixel, output_binary = self.deepPix.run(["output_pixel", "output_binary"],
                                                       {"input": face_tensor.astype(np.float32)})
        liveness_score = (np.mean(output_pixel.flatten()) +
                          np.mean(output_binary.flatten()))/2.0
        return liveness_score


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize(
            (image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    return face

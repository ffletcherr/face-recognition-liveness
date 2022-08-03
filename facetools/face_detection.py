from typing import List, Tuple

import mediapipe as mp
import numpy as np

from .utils import extract_face


class FaceDetection:
    def __init__(self, max_num_faces: int = 1):
        self.detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces, static_image_mode=True
        )

    def __call__(self, image) -> Tuple[List[np.ndarray], List[List[int]]]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        boxes = []
        faces = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array(
                    [(pt.x * w, pt.y * h) for pt in prediction.landmark],
                    dtype=np.float64,
                )
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                face_arr = extract_face(image, bbox.flatten().tolist())
                boxes.append(bbox)
                faces.append(face_arr)
        return faces, boxes

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime
import pandas as pd
from PIL import Image

from asset import detect_faces_mediapipe, extract_face, post_process, trans

th1 = .75
th2 = .95
detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, static_image_mode=True)
deepPix_onnx = onnxruntime.InferenceSession(
    "Pretrained_models/OULU_Protocol_2_model_0_0.onnx", providers=['CPUExecutionProvider'])
resnet_onnx = onnxruntime.InferenceSession(
    "Pretrained_models/InceptionResnetV1_vggface2.onnx", providers=['CPUExecutionProvider'])

facebank = pd.read_csv('face-ver-code/hm.csv', header=None)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bboxs = detect_faces_mediapipe(detector, frame)
    if not len(bboxs):
        continue
    bbox = bboxs[0]

    # calculate identification score
    face_arr = extract_face(frame, bbox.flatten())
    face_tensor = post_process(face_arr)
    embeddings = resnet_onnx.run(["output"],
                                 {"input": face_tensor.unsqueeze(0).detach().cpu().numpy().astype(np.float32)})[0]
    diff = facebank - embeddings
    norm = np.linalg.norm(diff, axis=1)
    min_verify_score = np.around(norm.min(), 3)
    mean_verify_score = np.around(norm.mean(), 3)

    # calculate liveness score
    face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = trans(face_pil)
    output_pixel, output_binary = deepPix_onnx.run(["output_pixel", "output_binary"],
                                                   {"input": face_tensor.unsqueeze(0).detach().cpu().numpy().astype(np.float32)})
    liveness_score = (np.mean(output_pixel.flatten()) +
                      np.mean(output_binary.flatten()))/2.0

    cv2.imshow("face", face_arr)
    k = cv2.waitKey(2)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

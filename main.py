import os
from pathlib import Path

import cv2

from modules import FaceDetection, IdentityVerification, LivenessDetection

root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"

resNet_checkpoint_path = data_folder / "InceptionResnetV1_vggface2.onnx"
facebank_path = data_folder / "hm.csv"

deepPix_checkpoint_path = data_folder / "OULU_Protocol_2_model_0_0.onnx"

faceDetector = FaceDetection()
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(), facebank_path=facebank_path.as_posix())
livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix())

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, boxes = faceDetector(frame)
    if not len(faces):
        continue

    face_arr = faces[0]
    min_sim_score, mean_sim_score = identityChecker(face_arr)
    liveness_score = livenessDetector(face_arr)

    cv2.imshow("face", face_arr)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

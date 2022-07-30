import os
from os import environ
from pathlib import Path

import cv2
import jsonpickle
import numpy as np
from dotenv import load_dotenv
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from flask import Flask, Response, request

root = Path(os.path.abspath(__file__)).parent.absolute()

load_dotenv((root / ".env").as_posix())  # take environment variables from .env.

data_folder = environ.get("DATA_FOLDER")
resnet_name = environ.get("RESNET")
deeppix_name = environ.get("DEEPPIX")
facebank_name = environ.get("FACEBANK")


data_folder = root.parent / data_folder

resNet_checkpoint_path = data_folder / "checkpoints" / resnet_name
facebank_path = data_folder / facebank_name

deepPix_checkpoint_path = data_folder / "checkpoints" / deeppix_name

faceDetector = FaceDetection()
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
)
livenessDetector = LivenessDetection(checkpoint_path=deepPix_checkpoint_path.as_posix())

app = Flask(__name__)


@app.route("/main", methods=["POST"])
def main():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "min_sim_score": None,
            "mean_sim_score": None,
            "liveness_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        response = {
            "message": "Everything is OK.",
            "min_sim_score": min_sim_score.item(),
            "mean_sim_score": mean_sim_score.item(),
            "liveness_score": liveness_score.item(),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )


@app.route("/identity", methods=["POST"])
def identity():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "min_sim_score": None,
            "mean_sim_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)

        response = {
            "message": "Everything is OK.",
            "min_sim_score": min_sim_score.item(),
            "mean_sim_score": mean_sim_score.item(),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )


@app.route("/liveness", methods=["POST"])
def liveness():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "liveness_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        response = {
            "message": "Everything is OK.",
            "liveness_score": liveness_score.item(),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )


if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000)

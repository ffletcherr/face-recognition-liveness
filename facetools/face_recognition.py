import urllib
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime
import pandas as pd
import progressbar

pbar = None

class IdentityVerification:
    def __init__(self, checkpoint_path: str, facebank_path: str):
        if not Path(checkpoint_path).is_file():
            print("Downloading the Inception resnet v1 onnx checkpoint")
            urllib.request.urlretrieve(
                "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/InceptionResnetV1_vggface2.onnx",
                Path(checkpoint_path).absolute().as_posix(), show_progress
            )
        if not Path(facebank_path).is_file():
            raise FileNotFoundError(
                f"{facebank_path} is not a file. Please check the path."
            )

        self.resnet = onnxruntime.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )
        self.facebank = pd.read_csv(facebank_path, header=None)

    def __call__(self, face_arr: np.ndarray) -> Tuple[float, float]:
        face_arr = np.moveaxis(face_arr, -1, 0)
        input_arr = np.expand_dims((face_arr - 127.5) / 128.0, 0)
        embeddings = self.resnet.run(
            ["output"], {"input": input_arr.astype(np.float32)}
        )[0]
        diff = self.facebank - embeddings
        norm = np.linalg.norm(diff, axis=1)
        min_sim_score = np.around(norm.min(), 3)
        mean_sim_score = np.around(norm.mean(), 3)
        return min_sim_score, mean_sim_score


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

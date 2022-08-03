# face-recognition-liveness

Face liveness detection and indentity recognition using fast and accurate convolutional neural networks is implemented in Pytorch. Also a Flask API and ready-to-use Dockerfile can be found in this repository.

This project uses [Mediapipe](https://github.com/google/mediapipe) for face detection and the face recognition model is borrowed from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) .The liveness detection model came from [Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection](https://arxiv.org/abs/1907.04047) paper and, [pre-trained models](https://www.idiap.ch/software/bob/docs/bob/bob.paper.deep_pix_bis_pad.icb2019/master/pix_bis.html#using-pretrained-models) are published by authors.


![face recognition and liveness](https://user-images.githubusercontent.com/43831412/181917410-a7df598b-8e89-419c-9505-6111676dc3a4.jpg)


## Getting Started

Download .onnx models and put them in `data/checkpoints` folder.

- [InceptionResnetV1_vggface2.onnx](https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/InceptionResnetV1_vggface2.onnx)
- [OULU_Protocol_2_model_0_0.onnx](https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/OULU_Protocol_2_model_0_0.onnx)

>*Note:* If you have an internet connection, models will be downloaded automatically.

## Simple Usage

Run the following command to check liveness (and test are you Ryan Reynolds or not!)

```bash
$ python webcam_test.py
```

>*Note:* Liveness score is between 0 and 1 and, in average, it is enough be greater than ~ **0.03** to be considered as a live image.


## Create Facebank CSV
In the first step you need a facebank. So put some images (jpg, jpeg, png) in a folder and create facebank csv file using `create_facebank.py` script:
```
$ python3 create_facebank.py --images ./data/images \
--checkpoint ./data/checkpoints/InceptionResnetV1_vggface2.onnx \
--output ./data/test.csv
```
--images: the path to the images folder

--checkpoint: the path to the resnet vggface2 onnx checkpoint

--output: the path to the output csv file

## Run Docker

Now you can start the deployment process. Variables (models and facebank names) can be changed in `app/.env` file:
```
DATA_FOLDER=data
RESNET=InceptionResnetV1_vggface2.onnx
DEEPPIX=OULU_Protocol_2_model_0_0.onnx
FACEBANK=test.csv
```


First build the docker image:

```
$ sudo docker build --tag face-demo .
```

Now run the image as a container:

```
$ sudo docker run -p 5000:5000 face-demo python3 -m flask run --host=0.0.0.0 --port=5000
```

### Test
Finally we can test our app using a python client. So for testing just run this:
```
# face-recognition-liveness/
$ cd ./app
$ python3 client.py --image ../data/images/reynolds_001.png --host localhost --port 5000 --service main 
```

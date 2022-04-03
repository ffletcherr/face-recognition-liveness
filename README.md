# face-recognition-liveness

First we must create a csv which contains each face embedding vector. Then we can build a docker image and run our app as a container.

## Create Facebank CSV
In the first step you need a facebank. So put some images (jpg, jpeg, png) in a folder and create facebank csv file using `create_facebank.py` script:
```
python3 create_facebank.py --images ./data --checkpoint ./data/InceptionResnetV1_vggface2.onnx --output ./data/test.csv
```
--images: the path to the images folder

--checkpoint: the path to the resnet vggface2 onnx checkpoint

--output: the path to the output csv file

## Run Docker

Now you can start the deployment process. Variables (models and facebank names) can be changed in `dot_env` file:
```
DATA_FOLDER=data
RESNET=InceptionResnetV1_vggface2.onnx
DEEPPIX=OULU_Protocol_2_model_0_0.onnx
FACEBANK=test.csv
```


First build the docker image:

```
sudo docker build --tag face-demo .
```

Now run the image as a container:

```
sudo docker run -p 5000:5000 face-demo python3 -m flask run --host=0.0.0.0 --port=5000
```

## Test
Finally we can test our app using a python client. So for testing just run this:
```
python3 client.py --image ./data/test.jpg --host localhost --port 5000 --service main 
```

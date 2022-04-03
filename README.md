# face-recognition-liveness



build docker:

```
sudo docker build --tag face-demo .
```

run image as a container:

```
sudo docker run -p 5000:5000 face-demo python3 -m flask run --host=0.0.0.0 --port=5000
```

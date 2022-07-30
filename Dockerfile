# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

RUN pip3 install -e .

WORKDIR /app/app

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]

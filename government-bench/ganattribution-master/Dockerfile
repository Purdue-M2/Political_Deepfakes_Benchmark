#ARG REGISTRY_OVERRIDE=docker.io
#FROM ${REGISTRY_OVERRIDE}/python:3.8
##################################################YY BEGIN SIREN DOCKER
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime 


RUN apt-get update
RUN apt-get install -y \
  sudo vim man less git wget bzip2 build-essential \
  ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1

#############################################################################
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
####################################################
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN apt-get install -y vim-tiny

RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN wget -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth  https://download.pytorch.org/models/resnet50-19c8e357.pth


WORKDIR /app
COPY . /app



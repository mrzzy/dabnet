#
# Dockerfile
# dabnet 
# dabnet container
#

ARG TF_IMG=tensorflow/tensorflow:1.13.1-gpu-py3
FROM $TF_IMG

## Install project dependencies 
# install package requirements
#RUN apt-get update && apt-get install -y 
#    && rm -rf /var/lib/apt/lists/*

# Install module requirements
COPY containers/dabnet/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

## Setup environment 
# python path 
ENV PYTHONPATH="deps/posenet/:src/:${PYTHONPATH}"

## Setup project directory
RUN mkdir /project
WORKDIR /project
# Copy source code
COPY src /project/src
# Copy models
COPY models /project/models


#
# Dockerfile
# dabnet 
# posenet container
#

FROM tensorflow/tensorflow:1.13.1-gpu-py3

## Install project dependencies 
# install package requirements
RUN apt-get update && apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install module requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# setup environment variables
COPY .env /tmp/.env
RUN bash -c 'echo "source /tmp/.env" >> /etc/bash.bashrc'

## Setup project directory
RUN mkdir /project
WORKDIR /project
# Copy dependencies
COPY deps /project/deps
# Copy source code
COPY src /project/src


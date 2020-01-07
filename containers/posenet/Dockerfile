#
# Dockerfile
# dabnet 
# posenet container
#

ARG TF_IMG=tensorflow/tensorflow:1.13.1-gpu-py3
FROM $TF_IMG

## Install project dependencies 
# install package requirements
RUN apt-get update && apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libcurl4-openssl-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install module requirements
COPY containers/posenet/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# setup environment variables
ENV PYTHONPATH="deps/posenet/:src/:${PYTHONPATH}"
## Setup project directory
RUN mkdir /project
WORKDIR /project
# Copy dependencies
COPY deps /project/deps
# Copy source code
COPY src /project/src
# Copy models
RUN mkdir -p  /project/models
COPY models/posenet /project/models/posenet

# Run posenet server
EXPOSE 8088
CMD python3 src/pose/server.py

#
# Dockerfile
# dabnet container
#

FROM tensorflow/tensorflow:2.0.0a0-py3

## Setup apt https suppport
RUN apt-get update && apt-get install -y apt-transport-https

## Install conda
# Add conda key
COPY conda.gpg /tmp/conda.gpg
RUN install -o root -g root -m 644 /tmp/conda.gpg /etc/apt/trusted.gpg.d/
# Add conda repo
RUN bash -c 'echo "deb [arch=amd64] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list'
# Install conda package
RUN apt-get update && apt-get install -y \
    conda
# remove package lists
RUN rm -rf  /var/lib/apt/lists/*

#
# Dockerfile
# dabnet 
# dabnet container
#

FROM python:3.6

## Install project dependencies
# install opencv dependencies
RUN apt-get update && apt-get install -y libopencv-dev python-opencv \
    && rm -rf /var/lib/apt/lists/

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
COPY models/model.pickle /project/models/
# copy dataset metadata
RUN mkdir data
COPY data/meta.csv data/

# run dabnet server
EXPOSE 5000
CMD python3 src/server.py

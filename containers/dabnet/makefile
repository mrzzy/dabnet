#
# dabnet 
# dabnet contaier makefile
#

CONTAINER:=dabnet-dabnet
USE_GPU:=1
DEBUG:=1
TAG:=devel

ifeq ($(USE_GPU),1)
RUNTIME:=nvidia
TF_IMG:=tensorflow/tensorflow:1.13.1-gpu-py3
else
TF_IMG:=tensorflow/tensorflow:1.13.1-py3
TAG:=$(TAG)-cpu
endif 

ifeq ($(DEBUG),1)
TF_IMG:=$(TF_IMG)-jupyter
endif

.PHONY: clean build rebuild purge run

build:: containers/$(CONTAINER).marker

clean::
	-rm -f containers/$(CONTAINER).marker

purge:: clean
	docker rmi -f $(CONTAINER)

containers/$(CONTAINER).marker:
	docker build -f containers/dabnet/Dockerfile . -t $(CONTAINER):$(TAG) \
		$(if $(TF_IMG),--build-arg TF_IMG="$(TF_IMG)",)
	touch $@

run: containers/$(CONTAINER).marker
	-docker run $(if $(RUNTIME),--runtime=$(RUNTIME),)\
		-p 8088:8088 \
		-it -v "$(CURDIR)/src:/project/src" $(CONTAINER)


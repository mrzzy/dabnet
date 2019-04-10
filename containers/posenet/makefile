#
# dabnet 
# pose container makefile
#

CONTAINER:=dabnet-posenet
USE_GPU:=1

ifeq ($(USE_GPU),1)
RUNTIME:=nvidia
TAG:=
else
RUNTIME:=
TF_IMG:=tensorflow/tensorflow:1.13.1-py3
TAG:=:cpu
endif 

.PHONY: clean build rebuild purge run

build:: containers/$(CONTAINER).marker

clean::
	-rm -f containers/$(CONTAINER).marker

purge:: clean
	docker rmi -f $(CONTAINER)

containers/$(CONTAINER).marker:
	docker build -f containers/posenet/Dockerfile . -t $(CONTAINER)$(TAG) \
		$(if $(TF_IMG),--build-arg TF_IMG="$(TF_IMG)",)
	touch $@

run: containers/$(CONTAINER).marker
	-docker run $(if $(RUNTIME),--runtime=$(RUNTIME),)\
		-p 8088:8088 \
		-it -v "$(CURDIR)/src:/project/src" $(CONTAINER)


#
# dabnet 
# project makefile
#

RUNTIME:=nvidia
CONTAINER:=dabnet
DOCKER_USERNAME:=mrzzy
.PHONY: all clean build rebuild purge run

all: build run

build: containers/$(CONTAINER).marker

clean: 
	rm -f containers/$(CONTAINER).marker

rebuild: clean build

purge:
	docker rmi -f $(CONTAINER)

containers/$(CONTAINER).marker:
	docker build -f containers/Dockerfile . -t $(CONTAINER)
	touch $@

# Docker
run: containers/$(CONTAINER).marker
	@ docker run $(if $(RUNTIME),--runtime=$(RUNTIME),)\
		-u $(shell id -u):$(shell id -g) \
		-it -v "$(CURDIR):/project" $(CONTAINER)


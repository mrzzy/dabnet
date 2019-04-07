#
# dabnet 
# project makefile
#

CONTAINER:=container/dabnet
DOCKER_USERNAME:=mrzzy
.PHONY: all clean build run

all: build run

build: $(CONTAINER)

$(CONTAINER):
	docker build -f containers/Dockerfile . -t $(notdir $@)
	touch $@

run: $(CONTAINER)
	docker run -it $(notdir $@)

clean: 
	rm -f $(CONTAINER)

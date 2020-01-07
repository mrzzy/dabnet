#
# dabnet 
# project wide makefile
#

.PHONY: all build run push

all: build 

build:
	docker-compose build

run:
	docker-compose up

push:
	docke-compose push

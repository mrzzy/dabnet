#
# dabnet 
# pose container makefile
#

RUNTIME:=nvidia
CONTAINER:=dabnet-posenet

.PHONY: clean build rebuild purge run

build:: containers/$(CONTAINER).marker

clean::
	-rm -f containers/$(CONTAINER).marker

purge:: clean
	docker rmi -f $(CONTAINER)

containers/$(CONTAINER).marker:
	docker build -f containers/posenet/Dockerfile . -t $(CONTAINER)
	touch $@

run: containers/$(CONTAINER).marker
	-docker run $(if $(RUNTIME),--runtime=$(RUNTIME),)\
		-p 8088:8088 \
		-it -v "$(CURDIR)/src:/project/src" $(CONTAINER)


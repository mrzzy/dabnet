#
# dabnet 
# pose container makefile
#

RUNTIME:=nvidia
CONTAINER:=dabnet-pose

.PHONY: clean build rebuild purge run

build:: containers/$(CONTAINER).marker

clean::
	-rm -f containers/$(CONTAINER).marker

purge::
	docker rmi -f $(CONTAINER)

containers/$(CONTAINER).marker:
	docker build -f src/pose/Dockerfile . -t $(CONTAINER)
	touch $@

# Docker
run: containers/$(CONTAINER).marker
	-docker run $(if $(RUNTIME),--runtime=$(RUNTIME),)\
		-u $(shell id -u):$(shell id -g) \
		-it -v "$(CURDIR):/project" $(CONTAINER)


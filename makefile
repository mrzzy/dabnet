#
# dabnet 
# project wide makefile
#

MAKEFILES:=containers/posenet/makefile
.PHONY: all clean build rebuild 

all: build 

rebuild: clean build

include $(MAKEFILES)

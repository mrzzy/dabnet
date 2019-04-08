#
# dabnet 
# project wide makefile
#

MAKEFILES:=src/pose/makefile
.PHONY: all clean build rebuild 

all: build 

rebuild: clean build

include $(MAKEFILES)

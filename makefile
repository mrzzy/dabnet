#
# dabnet 
# project wide makefile
#

MAKEFILES:=src/pose/makefile
.PHONY: all clean build rebuild 

all: build 

build: 
	$(foreach MAKEFILE, $(MAKEFILES), $(MAKE) -f $(MAKEFILE) build)

clean: 
	$(foreach MAKEFILE, $(MAKEFILES), $(MAKE) -f $(MAKEFILE) clean)
	

rebuild: clean build

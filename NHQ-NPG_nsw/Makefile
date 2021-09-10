GO_BIN := ${shell which go}
CURRENT_DIR := ${PWD}
MAJOR := 0
MINOR := 1
MICRO := 4
VERSION := $(MAJOR).$(MINOR).$(MICRO)
PREFIX ?= /usr/local/
PREFIX_INCLUDE := $(PREFIX)/include
PREFIX := $(PREFIX)/lib64
CXX ?= g++

all: static_lib shared_lib 
### Bindings ### 

### Libraries ###
shared_lib:
	cd src/ && make shared_lib && cd .. && mkdir -p build/lib && \
		mv src/libn2.so ./build/lib/libn2.so && \
		cp build/lib/libn2.so build/lib/libn2.so.$(VERSION)

static_lib:
	cd src/ && make static_lib && cd .. && mkdir -p build/lib/static && \
		mv src/libn2.a ./build/lib/static/

### Installation ###

install:
	if [ -a build/lib/libn2.so.$(VERSION) ] ; \
	then \
		install build/lib/libn2.so.$(VERSION) $(PREFIX) && \
		ln -s `which $(PREFIX)/libn2.so.$(VERSION)` $(PREFIX)/libn2.so.tmp && \
		mv $(PREFIX)/libn2.so.tmp $(PREFIX)/libn2.so ; \
		cp -r include/n2 $(PREFIX_INCLUDE) ; \
	fi;
	echo "Finished"



.PHONY: clean
clean:
	cd src && make clean
	rm -rf *.o
	rm -rf test_run
	rm -rf build

NVCC    ?= nvcc
ARCH    ?= sm_90
NVFLAGS := -O3 -arch=$(ARCH)

SRCS := $(wildcard src/*.cu)
BINS := $(patsubst src/%.cu,bin/%,$(SRCS))

all: $(BINS)

bin/%: src/%.cu | bin
	$(NVCC) $(NVFLAGS) $< -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin

.PHONY: all clean

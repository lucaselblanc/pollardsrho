TARGET    := pollardsrho
CXX       := g++
NVCC      := nvcc

CUDA_HOME ?= $(or $(shell echo $$HOME/cuda-13.0),/usr/local/cuda)

INCLUDES  := -I$(CUDA_HOME)/include
CXXFLAGS  := -O3 -march=native -Wall -std=c++14 -pthread $(INCLUDES)

FIRST_ARCH := $(firstword $(shell $(NVCC) --list-gpu-arch | grep -Eo 'compute_[0-9]+'))

NVCCFLAGS := -O3 -gencode arch=$(FIRST_ARCH),code=sm_$(subst compute_,,$(FIRST_ARCH)) \
             -ccbin $(CXX) \
             -Xcompiler "-O3 -std=c++14 -pthread" \
             $(INCLUDES) --expt-relaxed-constexpr

LDFLAGS   := -L$(CUDA_HOME)/lib64
LDLIBS    := -lcudart -lpthread

SRC_CPP   := pollardsrho.cpp
SRC_CU    := secp256k1.cu
OBJ       := $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(NVCC) --x cu $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ)
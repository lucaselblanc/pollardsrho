TARGET    := pollardsrho
CXX       := g++
NVCC      := nvcc

CUDA_HOME ?= $(or $(shell echo $$HOME/cuda-13.0),/usr/local/cuda)

INCLUDES  := -I$(CUDA_HOME)/include
CXXFLAGS  := -O3 -march=native -Wall -std=c++14 -pthread $(INCLUDES)

SUPPORTED_ARCHS := $(shell $(NVCC) --list-gpu-arch | grep -Eo 'compute_[0-9]+')

NVCCFLAGS := -O3 $(foreach arch,$(SUPPORTED_ARCHS),-gencode arch=$(arch),code=sm_$(subst compute_,,$(arch))) \
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
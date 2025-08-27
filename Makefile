TARGET    := pollardsrho
CXX       := g++
NVCC      := nvcc

CUDA_HOME ?= $(or $(shell echo $$HOME/cuda-13.0),/usr/local/cuda)

INCLUDES  := -I$(CUDA_HOME)/include
CXXFLAGS  := -O3 -march=native -Wall -std=c++14 -pthread $(INCLUDES)

NVCCFLAGS := -O3 \
    -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_35,code=compute_35 \  # PTX fallback
    -gencode arch=compute_50,code=compute_50 \
    -gencode arch=compute_60,code=compute_60 \
    -gencode arch=compute_70,code=compute_70 \
    -gencode arch=compute_75,code=compute_75 \
    -gencode arch=compute_80,code=compute_80 \
    -gencode arch=compute_86,code=compute_86 \
    -gencode arch=compute_89,code=compute_89 \
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
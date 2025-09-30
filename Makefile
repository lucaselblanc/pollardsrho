TARGET    := pollardsrho
CXX       := g++
NVCC      := nvcc

CUDA_HOME ?= $(or $(firstword $(wildcard $(HOME)/cuda-*)), /usr/local/cuda)

INCLUDES  := -I$(CUDA_HOME)/include
CXXFLAGS  := -O3 -march=native -Wall -std=c++14 -pthread $(INCLUDES)
LDFLAGS   := -L$(CUDA_HOME)/lib64
LDLIBS    := -lcudart -lpthread

SRC_CPP   := pollardsrho.cpp
SRC_CU    := secp256k1.cu
OBJ       := $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o)

.PHONY: all clean gpu_arch recurse

all: gpu_arch
	@$(MAKE) recurse

arch: arch.cu
	$(NVCC) arch.cu -o arch

gpu_arch: arch
	@./arch > gpu_arch 2>/dev/null || echo "GPU_ARCH := 0" > gpu_arch

recurse: $(TARGET)

-include gpu_arch

ifeq ($(GPU_ARCH),0)
	CXXFLAGS  := -O3 -march=native -Wall -std=c++14 -pthread
	NVCCFLAGS = -O3 -G -g \
	-std=c++14 \
	-ccbin $(CXX) \
	-Xcompiler "-O3 -pthread -fpermissive" \
	--expt-relaxed-constexpr
else
	NVCCFLAGS = -O3 -G -g \
	-std=c++14 \
	-gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) \
	-ccbin $(CXX) \
	-Xcompiler "-O3 -pthread -fpermissive" \
	$(INCLUDES) \
	--expt-relaxed-constexpr
endif

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJ) arch gpu_arch
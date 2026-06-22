TARGET    := pollardsrho
CXX       := g++

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

export PATH := $(CUDA_HOME)/bin:$(CUDA_HOME)/nvvm/bin:$(PATH)

INCLUDES  := -I$(CUDA_HOME)/include -I$(CUDA_HOME)
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)
LDLIBS    := -lcudadevrt -lcudart_static -lpthread -ldl -lrt -lcrypto

SRC_CPP   := almostinverse.cpp
SRC_CU    := pollardsrho.cu secp256k1.cu
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ_CU    := $(SRC_CU:.cu=.o)
OBJ       := $(OBJ_CPP) $(OBJ_CU)

.PHONY: all fresh recurse clean

all: fresh

fresh: gpu_arch
	@$(MAKE) recurse

gpu_arch: arch
	@RESULT=$$(./arch 2>/dev/null | grep -E '^[0-9]+$$' || echo "0"); \
	echo "GPU_ARCH := $$RESULT" > gpu_arch

arch: arch.cu
	$(NVCC) $(INCLUDES) $(LDFLAGS) -ccbin $(CXX) arch.cu -o arch $(LDLIBS)

recurse: $(TARGET)

-include gpu_arch

CXXFLAGS  := -g -O3 -std=c++14 -pthread -I. $(INCLUDES)
NVCCFLAGS := -g -G -O3 -std=c++14 -rdc=true -dc -ccbin $(CXX) $(INCLUDES) \
             -Xcompiler "-g -O3 -pthread -fpermissive -fPIC" \
             --expt-relaxed-constexpr --maxrregcount=96 -MD

DLINKFLAGS :=
ifneq ($(filter-out 0,$(strip $(GPU_ARCH))),)
    NVCCFLAGS += -gencode arch=compute_$(strip $(GPU_ARCH)),code=sm_$(strip $(GPU_ARCH))
    DLINKFLAGS := -gencode arch=compute_$(strip $(GPU_ARCH)),code=sm_$(strip $(GPU_ARCH))
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

dlink.o: $(OBJ_CU)
	$(NVCC) $(DLINKFLAGS) $(INCLUDES) -rdc=true -dlink $(OBJ_CU) -o dlink.o $(LDFLAGS) -ccbin $(CXX)

$(TARGET): $(OBJ) dlink.o
	$(NVCC) $(DLINKFLAGS) $(OBJ) dlink.o -o $@ $(LDFLAGS) $(LDLIBS) -ccbin $(CXX) -Xcompiler "-I$(CUDA_HOME)/include -I$(CUDA_HOME)"

clean:
	@echo "Cleaning..."
	rm -f pollardsrho arch gpu_arch dlink.o
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
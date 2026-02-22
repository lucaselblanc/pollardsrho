TARGET    := pollardsrho
CXX       := g++

ARCH_NAME := $(shell uname -m)

ifeq ($(ARCH_NAME), x86_64)
    CUDA_DIR_NAME := cuda_x86_64
else ifeq ($(ARCH_NAME), aarch64)
    CUDA_DIR_NAME := cuda_aarch64
else
    $(error Arch $(ARCH_NAME) does not support!)
endif

CUDA_HOME := $(abspath $(CURDIR)/$(CUDA_DIR_NAME))
NVCC      := $(CUDA_HOME)/bin/nvcc

export PATH := $(CUDA_HOME)/bin:$(CUDA_HOME)/nvvm/bin:$(PATH)

INCLUDES  := -I$(CUDA_HOME)/include -I$(CUDA_HOME)/bin
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)
LDLIBS    := -lcudart_static -lcudadevrt -lpthread -ldl -lrt
SRC_CPP   := pollardsrho.cpp almostinverse.cpp
SRC_CU    := secp256k1.cu
OBJ       := $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o)

.PHONY: all clean gpu_arch recurse set_perms

all: set_perms gpu_arch
	@$(MAKE) recurse

set_perms:
	@echo "Configuring CUDA Toolkit Permissions..."
	@find $(CUDA_HOME) -type f -path "*/bin/*" -exec chmod +x {} + 2>/dev/null || true

arch: arch.cu
	$(NVCC) $(INCLUDES) $(LDFLAGS) -ccbin $(CXX) arch.cu -o arch $(LDLIBS)

gpu_arch: arch
	@./arch > gpu_arch 2>/dev/null || echo "GPU_ARCH := 0" > gpu_arch

recurse: $(TARGET)

-include gpu_arch

NVCCFLAGS := -O3 -std=c++14 -ccbin $(CXX) $(INCLUDES) \
             -Xcompiler "-O3 -pthread -fpermissive -I$(CUDA_HOME)/include" \
             --expt-relaxed-constexpr -MD

ifneq ($(GPU_ARCH),0)
    NVCCFLAGS += -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
endif

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS) -Xcompiler "-I$(CUDA_HOME)/include"

clean:
	rm -f $(TARGET) $(OBJ) arch gpu_arch *.d
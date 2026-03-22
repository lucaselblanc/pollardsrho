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

LEGACY_V1 ?= 0
ifeq ($(firstword $(MAKECMDGOALS)), LEGACY)
    LEGACY_V1 := 1
endif

CUDA_HOME := $(abspath $(CURDIR)/$(CUDA_DIR_NAME))
NVCC      := $(CUDA_HOME)/bin/nvcc

export PATH := $(CUDA_HOME)/bin:$(CUDA_HOME)/nvvm/bin:$(PATH)
export LEGACY_V1

ifeq ($(LEGACY_V1), 1)
    SRC_CPP := pollardsrho_legacy.cpp almostinverse.cpp
else
    SRC_CPP := pollardsrho.cpp almostinverse.cpp
endif

INCLUDES  := -I$(CUDA_HOME)/include -I$(CUDA_HOME)/bin
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)
LDLIBS    := -lcudart_static -lcudadevrt -lpthread -ldl -lrt
SRC_CU    := secp256k1.cu
OBJ       := $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o)

.PHONY: all clean gpu_arch recurse set_perms LEGACY

LEGACY: clean all

all: gpu_arch set_perms
	@$(MAKE) recurse

set_perms:
	@echo "Configuring CUDA Toolkit Permissions..."
	@find $(CUDA_HOME) -type f -path "*/bin/*" -exec chmod +x {} + 2>/dev/null || true

arch: arch.cu
	$(NVCC) $(INCLUDES) $(LDFLAGS) -ccbin $(CXX) arch.cu -o arch $(LDLIBS)

gpu_arch: arch
	@./arch 2>/dev/null | grep -E '^[0-9]+$$' > .temp_arch || echo "0" > .temp_arch
	@echo "GPU_ARCH := $$(cat .temp_arch)" > gpu_arch
	@rm -f .temp_arch

recurse: $(TARGET)

ifneq ($(MAKECMDGOALS),clean)
-include gpu_arch
endif

NVCCFLAGS := -O3 -std=c++14 -ccbin $(CXX) $(INCLUDES) \
             -Xcompiler "-O3 -pthread -fpermissive -I$(CUDA_HOME)/include" \
             --expt-relaxed-constexpr -MD

ifneq ($(strip $(GPU_ARCH)),)
ifneq ($(strip $(GPU_ARCH)),0)
	NVCCFLAGS += -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
endif
endif

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS) -Xcompiler "-I$(CUDA_HOME)/include"

clean:
	@echo "Cleaning..."
	rm -f pollardsrho arch gpu_arch *.o *.d

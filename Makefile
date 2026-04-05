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
LDLIBS    := -lcudart_static -lcudadevrt -lpthread -ldl -lrt -lcrypto
SRC_CPP   := almostinverse.cpp
SRC_CU    := pollardsrho.cu secp256k1.cu
OBJ       := $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o)

.PHONY: all set_perms recurse gpu_arch

all: set_perms gpu_arch
	@$(MAKE) recurse

set_perms:
	@echo "Configuring CUDA Toolkit Permissions..."
	@find $(CUDA_HOME) -type f -path "*/bin/*" -exec chmod +x {} + 2>/dev/null || true

arch: arch.cu | set_perms
	$(NVCC) $(INCLUDES) $(LDFLAGS) -ccbin $(CXX) arch.cu -o arch $(LDLIBS)

gpu_arch: arch
	@RESULT=$$(./arch 2>/dev/null | grep -E '^[0-9]+$$' || echo "0"); \
	echo "GPU_ARCH := $$RESULT" > gpu_arch

recurse: $(TARGET)

-include gpu_arch

CXXFLAGS  := -O3 -std=c++14 -pthread -I. $(INCLUDES)
NVCCFLAGS := -O3 -std=c++14 -rdc=true -ccbin $(CXX) $(INCLUDES) \
             -Xcompiler "-O3 -pthread -fpermissive -I$(CUDA_HOME)/include" \
             --expt-relaxed-constexpr -MD

ifneq ($(filter-out 0,$(strip $(GPU_ARCH))),)
    NVCCFLAGS += -gencode arch=compute_$(strip $(GPU_ARCH)),code=sm_$(strip $(GPU_ARCH))
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LDFLAGS) -Xcompiler "$(CXXFLAGS)" $(LDLIBS)

clean:
	@echo "Cleaning..."
	rm -f pollardsrho arch gpu_arch *.o *.d

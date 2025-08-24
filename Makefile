TARGET    := pollardsrho
CXX       := g++-12
NVCC      := nvcc

CUDA_HOME ?= $(or $(shell echo $$HOME/cuda-13.0),/usr/local/cuda)

ifeq ($(wildcard $(CUDA_HOME)/bin/nvcc),)
  HAS_CUDA := 0
else
  HAS_CUDA := 1 #Substitua por: HAS_CUDA := 0 caso não houver hardware cuda, é útil somente para compilação.
endif

ifeq ($(HAS_CUDA),1)
  CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
                 awk -F. '{printf "sm_%d%d", $$1,$$2}' || echo "sm_75")
  ifeq ($(CUDA_ARCH),)
    CUDA_ARCH := sm_75
  endif
else
  CUDA_ARCH := sm_75
endif

INCLUDES  := -I$(CUDA_HOME)/include
CXXFLAGS  := -O3 -march=native -Wall -std=c++17 -pthread $(INCLUDES)
NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) -ccbin $(CXX) \
             -Xcompiler "-O3 -std=c++17 -pthread" \
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

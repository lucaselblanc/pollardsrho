#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No NVIDIA GPUs were detected, compiling for CPU...\n");
        printf("%d\n", 75);
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("%d%d\n", prop.major, prop.minor);
    return 0;
}
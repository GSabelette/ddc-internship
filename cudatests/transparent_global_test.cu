#include <stdio.h>
#include <cstdint>
#include "transparent_global.hpp"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

template <class Tag>
__global__ void iota_kernel(int* input, uint64_t size) {
    uint64_t thid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint64_t i = thid; i < size; i+= gridDim.x * blockDim.x) {
        input[i] = transparent_global<Tag, int>::get();
    }
}

class myTag;

int main() {
    int cst = 3;
    transparent_global<myTag, int>::init(cst);
    cudaCheckErrors("transparent_global initialization");

    uint64_t size = 10;
    int* input = nullptr;
    cudaMalloc((void**)&input, sizeof(int) * size);
    cudaCheckErrors("Array cudaMalloc");

    iota_kernel<myTag><<<3,3>>>(input, size);
    cudaCheckErrors("Kernel");
    
    cudaDeviceSynchronize();

    int ret[size];
    
    cudaMemcpy(&ret, input, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy Device to Host");
    for (int i = 0; i < size; ++i) printf("Ret %d : %d\n", i, ret[i]);
}
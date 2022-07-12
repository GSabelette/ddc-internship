#include <stdio.h>

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

__constant__ __device__ int* ptr;

__global__ void kernel() {
    printf("&ptr : %p\n\n", &ptr);
}

int main() {
    void* dev_addr = nullptr;

    kernel<<<1,1>>>();
    cudaCheckErrors("Kernel");
    cudaDeviceSynchronize();
    
    cudaGetSymbolAddress((void**)&dev_addr, ptr);
    cudaCheckErrors("getsymboladd");
    printf("dev addr : %p", dev_addr);
}
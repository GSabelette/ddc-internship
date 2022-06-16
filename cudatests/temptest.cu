#include <stdio.h>
#include <cstdint>

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

struct foo {
    int* a;
};

__global__ void kernel(foo* f) {
    printf("f->a : %d\n", *(f->a));
}

int main(int argc, char** argv) {
    foo* f = nullptr;
    int bar = 3;
    cudaMalloc((void**)&f, sizeof(foo));
    cudaCheckErrors("alloc");
    cudaMalloc((void**)&f->a, sizeof(int));
    cudaCheckErrors("alloc2");

    cudaMemcpy((&f->a), &bar, sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy");
    kernel<<<1,1>>>(f);
    cudaCheckErrors("kernel");
    cudaDeviceSynchronize();
}
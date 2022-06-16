#include <stdio.h>
#include <string>

__global__ void kernel(int* a) {
    printf("a : %p\n", a);
    printf("*a : %d\n", *a);
}

int main() {
    auto obj2 = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&obj2, sizeof(int));
    new(obj2) int(2);
    kernel<<<1,1>>>(obj2);
    cudaDeviceSynchronize();
    return 0;
}
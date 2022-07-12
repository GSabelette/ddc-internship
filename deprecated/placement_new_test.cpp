#include "deepcopy.hpp"
#include <stdio.h>

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;
using view = Kokkos::View<int*, DestSpace>;

__global__ void kernel(view* v) {
    printf("in kernel\n");
    new(v) view("", 10);
    for (int i = 0; i < v->extent(0); ++i) (*v)(i) = i;
    for (int i = 0; i < v->extent(0); ++i) printf("%d", (*v)(i));
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    view* v = new view();
    printf("pre kernel\n");
    kernel<<<1,1>>>(v);
    cudaDeviceSynchronize();
    return 0;
}

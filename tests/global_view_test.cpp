#include <stdio.h>

#include "kokkos_transparent_global.hpp"

#define VIEW_SIZE 10

class viewTag;

using global_kokkos = transparent_global<viewTag, Kokkos::View<int[VIEW_SIZE], Kokkos::LayoutRight>>;

int main(int argc, char** argv) {

    Kokkos::View<int[VIEW_SIZE], Kokkos::LayoutRight, Kokkos::CudaSpace> v;
    typename Kokkos::View<int[VIEW_SIZE], Kokkos::LayoutRight, Kokkos::CudaSpace>::HostMirror h_v;
    for (int i = 0; i < VIEW_SIZE; ++i) h_v(i) = i;
    Kokkos::deep_copy(v, h_v);
    global_kokkos::init(&v);
    Kokkos::parallel_for("Test for", Kokkos::RangePolicy<Kokkos::CudaSpace::execution_space>(0,10), KOKKOS_LAMBDA(int i) {
        printf("%d : %d", i, global_kokkos::get()(i));
    });
}
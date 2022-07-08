#include <stdio.h>

#include "global_var.hpp"

#define VIEW_SIZE 10

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;

class viewTag;

template <class ViewSpace>
using view = DeepcopyableView<int[VIEW_SIZE], ViewSpace>;

using global_view = global_template_var<viewTag, view>;

int main(int argc, char** argv) {
    view<SrcSpace> src_view;
    for (int i = 0; i < VIEW_SIZE; ++i) src_view(i) = i;

    global_view::init<view<DestSpace>, view<SrcSpace>>(src_view);

    Kokkos::parallel_for("Test for", Kokkos::RangePolicy<Kokkos::CudaSpace::execution_space>(0,10), KOKKOS_LAMBDA(int i) {
        printf("%d : %d", i, global_view::get()(i));
    });
}
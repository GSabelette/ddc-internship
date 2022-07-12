#include <stdio.h>

#include "global_var.hpp"

#define VIEW_SIZE 10

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;

class myTag;

template <class ViewSpace>
using view = Deepcopy::View<int*, ViewSpace>;

template <class MemorySpace>
struct myStruct {
    view<MemorySpace> v;

    myStruct() = default;

    explicit myStruct(std::size_t n) : v(Deepcopy::view_alloc<MemorySpace, int>(n * sizeof(int)), n)
    {
        fill_view(v, [&n] (auto v) {
            for (int i = 0; i < n; ++i) v(i) = i;
        });
    }

    template <class SrcSpace>
    myStruct(const myStruct<SrcSpace> src) : v(Deepcopy::view_alloc<MemorySpace>(src.v), src.v.extent(0)) {
        deepcopy<MemorySpace>(&v, src.v);
    }
};

using global_view = global_template_var<myTag, myStruct>;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    myStruct<SrcSpace> mystruct(10);

    global_view::init(mystruct);

    printf("CPU : ");
    for (int i = 0; i < VIEW_SIZE; ++i) printf("%d", global_view::get()->v(i));
    printf("\n\n");

    printf("GPU : ");
    Kokkos::parallel_for("Test for", Kokkos::RangePolicy<Kokkos::CudaSpace::execution_space>(0,1), KOKKOS_LAMBDA(int i) {
        for (int i = 0; i < VIEW_SIZE; ++i) printf("%d", global_view::get()->v(i));
        printf("\n");
    });
    Kokkos::fence();
}
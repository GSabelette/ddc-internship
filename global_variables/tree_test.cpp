#include "deepcopy.hpp"
#include <stdio.h>

template <class MemorySpace>
struct SimpleStruct {
    template <class ViewSpace>
    using view = Deepcopy::View<int*, ViewSpace>;

    view<MemorySpace> v;

    SimpleStruct(std::size_t n) 
        : v(Deepcopy::view_alloc<MemorySpace, int>(n*sizeof(int)), n)
    {
        fill_view(v, [&n](auto view) {
            for (int i = 0; i < n; ++i) view(i) = i;
        });
    }
    
    template <class SrcSpace>
    SimpleStruct(const SimpleStruct<SrcSpace> src) {
        deepcopy<MemorySpace, int*>(&v, src.v);
    }
};

template <class MemorySpace>
struct SomeStruct {
    SimpleStruct<MemorySpace>* s;
    SimpleStruct<MemorySpace> r_s;
    int* i;

    SomeStruct() : r_s(2)
    {
        init_alloc<MemorySpace>(&s, 1);
        init_alloc<MemorySpace>(&i, 2);
    }

    template <class SrcSpace>
    SomeStruct(const SomeStruct<SrcSpace> src) {
        deepcopy<MemorySpace, SimpleStruct<MemorySpace>*, SimpleStruct<SrcSpace>*>(&s, src.s);
        deepcopy<MemorySpace, SimpleStruct<MemorySpace>, SimpleStruct<SrcSpace>>(&r_s, src.r_s);
        deepcopy<MemorySpace, int*>(&i, src.i);
    }
};

template <class MemorySpace>
struct ComplexStruct {
    SomeStruct<MemorySpace>* s;
    int* i;

    ComplexStruct() 
    {
        init_alloc<MemorySpace>(&s);
        init_alloc<MemorySpace>(&i, 3);
    }

    template <class SrcSpace>
    ComplexStruct(ComplexStruct<SrcSpace> src) {
        deepcopy<MemorySpace, SomeStruct<MemorySpace>*, SomeStruct<SrcSpace>*>(&s, src.s);
        deepcopy<MemorySpace, int*>(&i, src.i);
    }
};

using DestSpace = Kokkos::CudaSpace;
using SrcSpace = Kokkos::HostSpace;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    ComplexStruct<DestSpace> cstruct;
    //printf("cstruct : %d | %d | %d | %d\n", *cstruct.i, *cstruct.s->i, cstruct.s->s->v(0), cstruct.s->r_s.v(1));

    Deepcopy::print_tree();
    Deepcopy::print_roots();

    printf("\nFreeing cstruct.s : %p\n", cstruct.s);
    Deepcopy::tree_free(cstruct.s);

    Deepcopy::print_tree();
    Deepcopy::print_roots();
}



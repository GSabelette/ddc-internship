#include "deepcopy.hpp"
#include <stdio.h>

template <class MemorySpace>
struct SomeStruct {   
    template <class ViewSpace>
    using view = DeepcopyableView<int*, ViewSpace>;

    int a;
    double* b;
    view<MemorySpace> c;

    SomeStruct() = default;

    explicit SomeStruct(std::size_t n) 
        : a(n)
        , b(empty_alloc<MemorySpace, double>(n))
        , c(deepcopyable_view_alloc<MemorySpace, int>(n * sizeof(int)), n)
    {
        fill_array<MemorySpace>(b, [&n](auto array) {
            for (int i = 0; i < n; ++i) array[i] = 2*i;
        }, n * sizeof(double));
        fill_view(c, [&n](auto v) {
            for (int i = 0; i < n; ++i) v(i) = i;
        });
    }

    template <class SrcSpace>
    SomeStruct(const SomeStruct<SrcSpace> src) : c(deepcopyable_view_alloc<MemorySpace>(src.c), src.c.extent(0)) { //c(deepcopyable_view_alloc<MemorySpace>(src.c), src.c.extent(0)) {//
        deepcopy<MemorySpace, view<MemorySpace>, view<SrcSpace>>(&c, src.c);
        deepcopy<MemorySpace, double*>(&b, src.b, src.a);
        deepcopy<MemorySpace, int>(&a, src.a);
    }
};  

template <class MemorySpace> 
struct ComplexStruct {

    SomeStruct<MemorySpace> s;

    ComplexStruct() = default;

    explicit ComplexStruct(std::size_t n) : s(n) {}

    template <class SrcSpace>
    ComplexStruct(const ComplexStruct<SrcSpace> src) {
        deepcopy<MemorySpace, SomeStruct<MemorySpace>, SomeStruct<SrcSpace>>(&s, src.s);
    }
};

template <class MemorySpace>
__host__ __device__ void print_struct(const SomeStruct<MemorySpace>& s) {
    printf("a : %d\n", s.a);
    printf("b : "); for (int i = 0; i < s.a; ++i) printf("%f ,", s.b[i]);
    printf("\nc : "); for (int i = 0; i < s.a; ++i) printf("%d", s.c(i));
    printf("\n"); 
}

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    SomeStruct<SrcSpace> mystruct(10);
    printf("mystruct : ");
    print_struct(mystruct);
    printf("\n");

    ComplexStruct<SrcSpace> cstruct(10);
    printf("cstruct : ");
    print_struct(cstruct.s);
    printf("\n");

    SomeStruct<DestSpace> mycopystruct(mystruct);
    SomeStruct<DestSpace>* myclonestruct = clone<DestSpace, SomeStruct<DestSpace>, SomeStruct<SrcSpace>>(mystruct);

    ComplexStruct<DestSpace> copycstruct(cstruct);
    ComplexStruct<DestSpace>* clonecstruct = clone<DestSpace, ComplexStruct<DestSpace>, ComplexStruct<SrcSpace>>(cstruct);

    Kokkos::parallel_for("Print copystruct", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("mycopystruct : ");
        print_struct(mycopystruct);
        printf("\n");

        printf("myclonestruct : ");
        print_struct(*myclonestruct);
        printf("\n");

        printf("copycstruct : ");
        print_struct(copycstruct.s);
        printf("\n");

        printf("clonecstruct : ");
        print_struct(clonecstruct->s);
        printf("\n");
    });
    Kokkos::fence();

    printf("\nQueues state : \n");
    Deepcopy::print_queues();
    printf("\nFreeing mystruct.c.data()\n");
    Deepcopy::free<SrcSpace>(mystruct.c.data());
    Deepcopy::print_queues();
    printf("\nFreeing copycstruct.s.c.data()\n");
    Deepcopy::free<DestSpace>(copycstruct.s.c.data());
    Deepcopy::print_queues();
    printf("\nFreeing all\n");
    Deepcopy::clear();
    Deepcopy::print_queues();
    
    Kokkos::fence();
}
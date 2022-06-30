#include "deepcopy.hpp"
#include <stdio.h>

template <class MemorySpace>
struct SomeStruct {   
    template <class ViewSpace>
    using view = DeepcopyableView<int*, ViewSpace>;

    view<MemorySpace> c;

    SomeStruct() = default;

    explicit SomeStruct(std::size_t n)
    {
        c = deepcopyable_view<int*, MemorySpace>(n * sizeof(int), n);
        fill_view(c, [](auto v) {
            for (int i = 0; i < 10; ++i) v(i) = i;
        });
    }

    template <class SrcSpace>
    SomeStruct(const SomeStruct<SrcSpace> src) : c(deepcopyable_view_alloc<MemorySpace>(src.c), src.c.extent(0)) { //c(deepcopyable_view_alloc<MemorySpace>(src.c), src.c.extent(0)) {//
        deepcopy<MemorySpace, view<MemorySpace>, view<SrcSpace>>(&c, src.c);
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

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    SomeStruct<SrcSpace> mystruct(10);
    printf("mystruct : ");
    for (int i = 0; i < 10; ++i) printf("%d", mystruct.c(i));
    printf("\n");

    ComplexStruct<SrcSpace> cstruct(10);
    printf("cstruct : ");
    for (int i = 0; i < 10; ++i) printf("%d", cstruct.s.c(i));
    printf("\n");

    SomeStruct<DestSpace> mycopystruct(mystruct);
    printf("pre clone\n");
    SomeStruct<DestSpace>* myclonestruct = clone<DestSpace, SomeStruct<DestSpace>, SomeStruct<SrcSpace>>(mystruct);

    ComplexStruct<DestSpace> copycstruct(cstruct);
    ComplexStruct<DestSpace>* clonecstruct = clone<DestSpace, ComplexStruct<DestSpace>, ComplexStruct<SrcSpace>>(cstruct);

    Kokkos::parallel_for("Print copystruct", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("copystruct : ");
        for (int i = 0; i < 10; ++i) printf("%d", mycopystruct.c(i));
        printf("\n");
        printf("clonestruct : ");
        for (int i = 0; i < 10; ++i) printf("%d", myclonestruct->c(i));
        printf("\n");

        printf("copycstruct : ");
        for (int i = 0; i < 10; ++i) printf("%d", copycstruct.s.c(i));
        printf("\n");
        printf("clonecstruct : ");
        for (int i = 0; i < 10; ++i) printf("%d", clonecstruct->s.c(i));
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
}
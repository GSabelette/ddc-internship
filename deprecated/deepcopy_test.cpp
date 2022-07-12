#include "deepcopy.hpp"
#include <stdio.h>

#define SOMESTRUCT 0
#define COMPLEXSTRUCT 1

template <class MemorySpace>
struct SomeStruct {   
    template <class ViewSpace>
    using view = Kokkos::View<double, ViewSpace>;

    int a;
    double* b;
    view<MemorySpace> c;

    SomeStruct() = default;

    explicit SomeStruct(std::size_t n)
        : a(n)
        , b(new double[n])
        , c("")
    {
        printf("SomeStruct constructor called with n = %d\n", n);
        printf("b : %p\n", b);
        for (int i = 0; i < n; ++i) b[i] = i;
        printf("helo\n");
        fill_view(c, [](auto v) {v() = 0.5;});
        printf("gudbi\n");
    }

    template <class SrcSpace>
    SomeStruct(const SomeStruct<SrcSpace> src) : c("") {
        deepcopy<MemorySpace, int>(&a, src.a);
        deepcopy<MemorySpace, double*>(&b, src.b, src.a);
        deepcopy<MemorySpace, view<Kokkos::CudaSpace>, view<Kokkos::HostSpace>>(&c, src.c);
    }

    // ~SomeStruct()
    // {
    //     printf("Destructor called\n");
    //     if (b)
    //     {
    //         delete [] b;
    //     }
    // }
};  

template <class MemorySpace>
struct ComplexStruct {
    SomeStruct<MemorySpace> s;

    ComplexStruct() : s(0) {};

    explicit ComplexStruct(std::size_t n) : s(n) {}

    template <class SrcSpace>
    ComplexStruct (const ComplexStruct<SrcSpace> src) : s(0) {
        deepcopy<MemorySpace, SomeStruct<MemorySpace>, SomeStruct<SrcSpace>>(&s, src.s);
    }
};

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    #if SOMESTRUCT == 1
    SomeStruct<SrcSpace> mystruct(3);
    printf("mystruct : %d, %f, %f, %f\n", mystruct.a, mystruct.b[0], mystruct.b[1], mystruct.c());

    SomeStruct<DestSpace> mycopystruct(mystruct);
    SomeStruct<DestSpace>* myclonestruct = clone<DestSpace, SomeStruct<DestSpace>, SomeStruct<SrcSpace>>(mystruct);

    Kokkos::parallel_for("Print copystruct", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("mycopystruct : %d, %f, %f, %f\n", mycopystruct.a, mycopystruct.b[0], mycopystruct.b[1], mycopystruct.c());
        printf("myclonestruct : %d, %f, %f, %f\n", myclonestruct->a, myclonestruct->b[0], myclonestruct->b[1], myclonestruct->c());
    });

    Kokkos::fence();
    #endif

    #if COMPLEXSTRUCT == 1
    ComplexStruct<SrcSpace> cstruct(3);
    printf("mycstruct : %d, %f, %f\n", cstruct.s.a, cstruct.s.b[1], cstruct.s.c());

    ComplexStruct<DestSpace> copycstruct(cstruct);
    //ComplexStruct<DestSpace>* myclonecstruct = clone<DestSpace, ComplexStruct<DestSpace>, ComplexStruct<SrcSpace>>(mycstruct);

    printf("entering parallel_for\n");
    Kokkos::parallel_for("Print copystruct", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("inside parallel for\n");
        printf("mycopycstruct :  %p\n", &copycstruct);
        //printf("mycopycstruct :  %d, %f, %f\n", copycstruct.s.a, copycstruct.s.b[1], copycstruct.s.c());
        //printf("myclonestruct : %d, %f, %f, %f\n", myclonestruct->a, myclonestruct->b[0], myclonestruct->b[1], myclonestruct->c());
    });

    printf("before fence\n");
    Kokkos::fence();
    printf("end endif\n");
    #endif
}
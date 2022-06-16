#include "deepcopy.hpp"
#include <stdio.h>

template <class MemorySpace>
struct SomeStruct
{   
    template <class ViewSpace>
    using view = Kokkos::View<double, ViewSpace>;

    int a;
    double* b;
    view<MemorySpace> c;

    explicit SomeStruct(std::size_t n)
        : a(n)
        , b(new double[n])
        , c("")
    {
        for (int i = 0; i < n; ++i) b[i] = i;
        c() = 0.5;
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

using DestSpace = Kokkos::CudaSpace;
using SrcSpace  = Kokkos::HostSpace;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    SomeStruct<SrcSpace> mystruct(3);
    printf("mystruct : %d, %f, %f, %f\n", mystruct.a, mystruct.b[0], mystruct.b[1], mystruct.c());

    SomeStruct<DestSpace> mycopystruct(mystruct);
    SomeStruct<DestSpace>* myclonestruct = clone<DestSpace, SomeStruct<DestSpace>, SomeStruct<SrcSpace>>(mystruct);

    Kokkos::parallel_for("Print copystruct", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("mycopystruct : %d, %f, %f, %f\n", mycopystruct.a, mycopystruct.b,mycopystruct.b[0], mycopystruct.b[1], mycopystruct.c());
        printf("myclonestruct : %d, %f, %f, %f\n", myclonestruct->a, myclonestruct->b[0], myclonestruct->b[1], myclonestruct->c());
    });

    Kokkos::fence();
}
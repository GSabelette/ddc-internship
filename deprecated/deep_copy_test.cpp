#include "deep_copy.hpp"

#define COMPLEX_STRUCT  1
#define KOKKOS_COPY     0

template <class MemorySpace>
struct complex_struct {
    int a;
    int* f_ptr = nullptr;
    Kokkos::View<int, MemorySpace> v;
    complex_struct() = delete;
    complex_struct(int v) : a(v) {}
    complex_struct(int v, int f) : a(v) {f_ptr = new int; *f_ptr = f; a = v;}

    template <class OriginSpace>
    complex_struct(const complex_struct<OriginSpace>& c) {
        this->a = c.a;
        this->f_ptr = empty_alloc<int, MemorySpace>();
        deepcopy<int>(f_ptr, *c.f_ptr);
    }

    template <class DestSpace = DefaultSpace>
    complex_struct<DestSpace>* clone() const {
        complex_struct<DestSpace>* clone = empty_alloc_destspace<complex_struct, DestSpace>();
        member_copy<int, DestSpace>(&clone->a, a);
        //std::cout << "&f_ptr : " << &f_ptr << "\n";
        member_copy<int*, DestSpace>(&clone->f_ptr, f_ptr);
        return clone;
    }
};

template <class MemorySpace>
struct complex_struct2 {
    complex_struct<MemorySpace>* f;
    complex_struct2(complex_struct<MemorySpace>* c) : f(c) {}

    template <class OriginSpace>
    complex_struct2(const complex_struct2<OriginSpace>& c) {
        complex_struct<MemorySpace> tmp(*c.f);
        f = empty_alloc_destspace<complex_struct, MemorySpace>();
        deepcopy<complex_struct<MemorySpace>>(f, tmp);
    }
    
    // A terme, constructeur par copie = appels a member_copy, clone = alloc + constructeur copie
    template <class DestSpace = DefaultSpace>
    complex_struct2<DestSpace>* clone() {
        complex_struct2<DestSpace>* clone = empty_alloc_destspace<complex_struct2, DestSpace>();
        member_copy<complex_struct<DestSpace>*, DestSpace>(&clone->f, f);
        //deepcopy<complex_struct<DestSpace>*, DestSpace>(&clone->f, f->template clone<DestSpace>());
        return clone;
    }
};

template <typename data_type, class MemorySpace>
struct view_struct {
    template <class Space>
    using view = Kokkos::View<data_type, Kokkos::LayoutLeft, Space>;
    template <class Space>
    using t_view_struct = view_struct<data_type, Space>;

    view<MemorySpace> v;

    template <class OriginSpace>
    view_struct(view<OriginSpace> ov) : v("view", ov.extent(0)) {Kokkos::deep_copy(v, ov);}

    template <class OriginSpace>
    view_struct(t_view_struct<OriginSpace> vs) : v("view", vs.v.extent(0)) {
        deepcopy<view, MemorySpace, OriginSpace>(&v, vs.v);
    }

    template <class DestSpace>
    t_view_struct<DestSpace>* clone() {
        t_view_struct<DestSpace>* clone = empty_alloc_destspace<t_view_struct, DestSpace>();
        t_view_struct<DestSpace> tmp_clone(*this);
        deepcopy<view<DestSpace>>(&clone->v, tmp_clone.v);
        return clone;
    }
};

template <typename data_type ,class MemorySpace>
using view = Kokkos::View<data_type, Kokkos::LayoutLeft, MemorySpace>;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    #if COMPLEX_STRUCT != 0
    complex_struct<Kokkos::HostSpace> host_cs(1,2);
    complex_struct2<Kokkos::HostSpace> host_cs2(&host_cs);
    printf("host_cs.a : %d\n", host_cs.a);
    printf("host_cs.f : %d\n", *(host_cs.f_ptr));

    complex_struct2<Kokkos::CudaSpace> device_cs2(host_cs2);

    printf("Cloning a complex_struct : \n");
    complex_struct<Kokkos::CudaSpace>* clone_cs = host_cs.clone<Kokkos::CudaSpace>();

    printf("Cloning a complex_struct2 : \n");
    complex_struct2<Kokkos::CudaSpace>* clone_cs2 = device_cs2.clone<Kokkos::CudaSpace>();

    Kokkos::parallel_for("Print Complex Struct", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("clone_cs->a : %d\n", clone_cs->a);
        printf("*clone_cs->f : %d\n", *(clone_cs->f_ptr)); 
        printf("clone_cs->f : %p\n", clone_cs->f_ptr);

        printf("device_cs2.f->a : %d\n", device_cs2.f->a);
        printf("*device_cs2.f->f : %d\n", *(device_cs2.f->f_ptr));
        printf("device_cs2.f->f : %p\n", (device_cs2.f->f_ptr));

        printf("clone_cs2.f->a : %d\n", clone_cs2->f->a);
        printf("*clone_cs2.f->f : %d\n", *(clone_cs2->f->f_ptr));
        printf("clone_cs2.f->f : %p\n", (clone_cs2->f->f_ptr));
    });
    Kokkos::fence();
    printf("\n");
    #endif 

    #if KOKKOS_COPY != 0
    int size = 1000000;
    view<int*, Kokkos::HostSpace> v("view 1", size);
    for (int i = 0; i < size; ++i) v(i) = 3;
    view<int*, Kokkos::HostSpace> v2(v);

    view_struct<int*, Kokkos::HostSpace> host_vs(v2);
    printf("host_vs : ");
    for (int i = 0; i < size; ++i) printf("%d", host_vs.v(i));
    printf("\n");

    view_struct<int*, Kokkos::HostSpace> host_vs2(host_vs);
    printf("host_vs2 : ");
    for (int i = 0; i < size; ++i) printf("%d", host_vs2.v(i));
    printf("\n");

    view_struct<int*, Kokkos::CudaSpace> cuda_vs(host_vs);

    view_struct<int*, Kokkos::CudaSpace>* cuda_clone_vs = host_vs2.clone<Kokkos::CudaSpace>();

    printf("Entering parallel region\n");
    Kokkos::parallel_for("Print Kokkos Copy", Kokkos::RangePolicy<Kokkos::Cuda>(0,1), KOKKOS_LAMBDA (const int& i) {
        printf("cuda_vs : ");
        //for (int i = 0; i < size; ++i) printf("%d", cuda_vs.v(i));
        printf("\n");

        printf("cuda_clone_vs : ");
        for (int i = 0; i < size; ++i) cuda_clone_vs->v(i) += 1;//printf("%d", cuda_clone_vs->v(i));
        printf("\n");
    });
    #endif
}
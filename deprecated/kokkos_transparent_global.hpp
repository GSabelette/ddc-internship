#include "deep_copy.hpp"
#include <type_traits>
#include <iostream>

#if defined (__CUDA_ARCH__)
using MemorySpace = Kokkos::CudaSpace;
#else
using MemorySpace = Kokkos::HostSpace;
#endif

template <typename T>
using enable_if_view_t = std::enable_if_t<Kokkos::is_view<T>::value, bool>;

namespace detail {
    namespace transparent_global_static {
        template<class Tag, typename cst_type>
        __constant__ __device__ cst_type device_value;

        template<class Tag, typename cst_type>
        static cst_type host_value;
    }
    namespace kokkos_transparent_global {
        template<class Tag, template <class> typename view_type>
        __constant__ __device__ view_type<Kokkos::CudaSpace> device_view;

        template<class Tag, template <class> typename view_type>
        static view_type<Kokkos::HostSpace> host_view;
    }
}

template <class Tag, typename cst_type>
class transparent_global {
    private :

    static __device__ __host__ cst_type& device_constant() {
        return detail::transparent_global_static::device_value<Tag, cst_type>;
    }

    static __host__ cst_type& host_constant() {
        return detail::transparent_global_static::host_value<Tag, cst_type>;
    }

    // fuse both using sfinae
    static __device__ __host__ cst_type<Kokkos::CudaSpace>& device_view_constant() {
        return detail::kokkos_transparent_global::device_view<Tag, cst_type>;
    }

    template <class sfinae_cst_type = cst_type,
              std::enable_if_t<is_same_template<cst_type, cst_type<Kokkos::CudaSpace>>::value, bool> = true>
    static __device__ __host__ cst_type<Kokkos::CudaSpace>& device_ct() {

    } 


    template <class sfinae_cst_type = cst_type,
              enable_if_view_t<sfinae_cst_type> = true>
    static __host__ cst_type<Kokkos::HostSpace>& host_view_constant() {
        return detail::kokkos_transparent_global::host_view<Tag, cst_type>;
    }

    public :
    __host__ __device__ transparent_global() = delete;

    template <class DestSpace,
              class SrcSpace>
    static void init(cst_type& val) {
        deepcopy<cst_type, SrcSpace>(host_constant(), *val);
        deepcopy<cst_type, DestSpace>(device_constant(), *val);
    }

    template <class DestSpace,
             class SrcSpace, 
             class sfinae_cst_type = cst_type,
             enable_if_view_t<sfinae_cst_type> = true
    > 
    static void init(Kokkos::View<SrcSpace>& val) {
        std::cout << "Called kokkos init method\n";
        deepcopy<cst_type<SrcSpace>>(&host_constant(), *val);
        deepcopy<cst_type<DestSpace>>(&device_constant(), *val);
    } 

    static __host__ __device__ cst_type& get() {
        #ifdef __CUDA_ARCH__
            return device_constant();
        #else
            return host_constant();
        #endif
    }
};

template <class Tag, template <class> typename view_type> 
class view_global {
    private :

    // fuse both using sfinae
    static __device__ __host__ cst_type<Kokkos::CudaSpace>& device_view_constant() {
        return detail::kokkos_transparent_global::device_view<Tag, cst_type>;
    }

    static __host__ cst_type<Kokkos::HostSpace>& host_view_constant() {
        return detail::kokkos_transparent_global::host_view<Tag, cst_type>;
    }

    public :
    __host__ __device__ transparent_global() = delete;

    template <class DestSpace,
              class SrcSpace>
    static void init(cst_type& val) {
        deepcopy<cst_type, SrcSpace>(host_constant(), *val);
        deepcopy<cst_type, DestSpace>(device_constant(), *val);
    }

    template <class DestSpace,
             class SrcSpace, 
             class sfinae_cst_type = cst_type,
             enable_if_view_t<sfinae_cst_type> = true
    > 
    static void init(Kokkos::View<SrcSpace>& val) {
        std::cout << "Called kokkos init method\n";
        deepcopy<cst_type<SrcSpace>>(&host_constant(), *val);
        deepcopy<cst_type<DestSpace>>(&device_constant(), *val);
    } 

    static __host__ __device__ cst_type& get() {
        #ifdef __CUDA_ARCH__
            return device_constant();
        #else
            return host_constant();
        #endif
    }
};
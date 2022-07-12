#include "deepcopy.hpp"
#include <type_traits>
#include <stdio.h>

template <typename T>
struct is_nonview_default_handled_type : std::integral_constant<bool, 
    is_default_handled_type_v<T> && !Kokkos::is_view<T>::value> {};

template <typename T>
inline constexpr bool is_nonview_default_handled_type_v = is_nonview_default_handled_type<T>::value;

template <typename T>
struct is_space_template_type : std::integral_constant<bool,
    !is_default_handled_type_v<T> || Kokkos::is_view<T>::value> {};

template <typename T>
inline constexpr bool is_space_template_type_v = is_space_template_type<T>::value;

// Global variables.
namespace detail {
    namespace transparent_global_static {
        template<class Tag, typename T>
        __constant__ __device__ T* device_value;

        template<class Tag, typename T>
        static T* host_value;
    }
    namespace transparent_global_template_static {
        template<class Tag, template <class> typename T>
        __constant__ __device__ T<Kokkos::CudaSpace>* device_value;

        template<class Tag, template <class> typename T>
        static T<Kokkos::HostSpace>* host_value;
    }
}

// For non-template types.
template <class Tag, typename T>
class global_var {

    private :
    static __host__ T* host_constant() {
        return detail::transparent_global_static::host_value<Tag, T>;
    }

    static __device__ __host__ T* device_constant() {
        return detail::transparent_global_static::device_value<Tag, T>;
    }

    public :
    __host__ __device__ global_var() = delete;

    template <typename T_dst = T, typename T_src = T,
              class DestSpace = Kokkos::CudaSpace, class SrcSpace = Kokkos::HostSpace>
    static void init(T_src& src) {
        deepcopy<SrcSpace, T_src*>(&host_constant(), &src);
        deepcopy<DestSpace, T_dst*, T_src*>(&device_constant(), &src);
    } 

    static __host__ __device__ T& get() {
        #ifdef __CUDA_ARCH__
            return *device_constant();
        #else
            return *host_constant();
        #endif
    }
};

// For space-template types.
template <class Tag, template<class> typename T>
class global_template_var {

    private :
    static __host__ T<Kokkos::HostSpace>** host_constant() {
        return &detail::transparent_global_template_static::host_value<Tag, T>;
    }

    static __device__ __host__ T<Kokkos::CudaSpace>** device_constant() {
        #ifdef __CUDA_ARCH__
            return &detail::transparent_global_template_static::device_value<Tag, T>;
        #else
            T<Kokkos::CudaSpace>** dev_value_address = nullptr;
            cudaGetSymbolAddress((void**)&dev_value_address, detail::transparent_global_template_static::device_value<Tag, T>);
            cudaCheckErrors("device_constant() : cudaGetSymbolAddress");
            return dev_value_address;
        #endif
    }

    public :
    __host__ __device__ global_template_var() = delete;

    template <typename T_dst = T<Kokkos::CudaSpace>, typename T_src = T<Kokkos::HostSpace>,
              class DestSpace = Kokkos::CudaSpace, class SrcSpace = Kokkos::HostSpace>
    static void init(T_src& src) {
        deepcopy<SrcSpace, T_src*>(host_constant(), &src);
        auto tmp = clone<DestSpace, T_dst>(src);
        cudaMemcpy((void**)device_constant(), &tmp, sizeof(long), cudaMemcpyDefault);
        cudaCheckErrors("init : cudaMemcpy");
    }

    template <typename U = T<CurrentSpace>>
    static __host__ __device__ U* get() {
        #ifdef __CUDA_ARCH__
            return *device_constant();
        #else
            return *host_constant();
        #endif
    }
};
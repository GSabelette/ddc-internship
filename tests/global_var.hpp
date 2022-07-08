#include "deepcopy.hpp"
#include <type_traits>
#include <stdio.h>

#if defined (__CUDA_ARCH__)
using CurrentSpace = Kokkos::CudaSpace;
#else
using CurrentSpace = Kokkos::HostSpace;
#endif

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

namespace detail {
    namespace transparent_global_static {
        template<class Tag, typename T>
        __constant__ __device__ T device_value;

        template<class Tag, typename T>
        static T host_value;
    }
    namespace transparent_global_template_static {
        template<class Tag, template <class> typename T>
        __constant__ __device__ T<Kokkos::CudaSpace> device_value;

        template<class Tag, template <class> typename T>
        static T<Kokkos::HostSpace> host_value;
    }
}

template <class Tag, typename T>
class global_var {

    private :
    // For non-template types
    template <std::enable_if_t<is_default_handled_type_v<T>, bool> = true,
              std::enable_if_t<!Kokkos::is_view<T>::value, bool> = true>
    static __host__ T& host_constant() {
        return detail::transparent_global_static::host_value<Tag, T>;
    }

    template <std::enable_if_t<is_default_handled_type_v<T>, bool> = true,
              std::enable_if_t<!Kokkos::is_view<T>::value, bool> = true>
    static __device__ __host__ T& device_constant() {
        return detail::transparent_global_static::device_value<Tag, T>;
    }

    // For template types
    // template <typename U = typename T<CurrentSpace>,
    //           std::enable_if_t<is_space_template_type_v<U>, bool> = true> 
    // static __host__ U& host_constant() {
    //     return detail::transparent_global_template_static::host_value<Tag, T>;
    // }

    // template <typename U = typename T<CurrentSpace>,
    //           std::enable_if_t<is_space_template_type_v<U>, bool> = true> 
    // static __device__ __host__ U& device_constant() {
    //     return detail::transparent_global_template_static::device_value<Tag, T>;
    // }

    public :
    __host__ __device__ global_var() = delete;

    template <typename T_dst = T, typename T_src = T_dst,
              class DestSpace = Kokkos::CudaSpace, class SrcSpace = Kokkos::HostSpace>
    static void init(T_src& src) {
        deepcopy<SrcSpace, T_src>(&host_constant(), src);
        deepcopy<DestSpace, T_dst, T_src>(&device_constant(), src);
    } 

    // For non-template types
    template <std::enable_if_t<is_default_handled_type_v<T>, bool> = true,
              std::enable_if_t<!Kokkos::is_view<T>::value, bool> = true>
    static __host__ __device__ T& get() {
        #ifdef __CUDA_ARCH__
            return device_constant();
        #else
            return host_constant();
        #endif
    }

    // For template types
    // template <typename U = T,
    //           std::enable_if_t<is_space_template_type_v<U>, bool> = true> 
    // static __host__ __device__ U<CurrentSpace>& get() {
    //     #ifdef __CUDA_ARCH__
    //         return device_constant();
    //     #else
    //         return host_constant();
    //     #endif
    // }
};

template <class Tag, template<class> typename T>
class global_template_var {

    private :
    // For template types
    template <std::enable_if_t<std::is_same_v<CurrentSpace, Kokkos::HostSpace>, bool> = true,
              std::enable_if_t<is_space_template_type_v<T<void>>, bool> = true> 
    static __host__ T<CurrentSpace>& host_constant() {
        return detail::transparent_global_template_static::host_value<Tag, T>;
    }

    template <std::enable_if_t<std::is_same_v<CurrentSpace, Kokkos::CudaSpace>, bool> = true,
              std::enable_if_t<is_space_template_type_v<T<void>>, bool> = true> 
    static __device__ __host__ T<CurrentSpace>& device_constant() {
        return detail::transparent_global_template_static::device_value<Tag, T>;
    }

    public :
    __host__ __device__ global_template_var() = delete;

    template <typename T_dst = T<Kokkos::CudaSpace>, typename T_src = T<Kokkos::HostSpace>,
              class DestSpace = Kokkos::CudaSpace, class SrcSpace = Kokkos::HostSpace>
    static void init(T_src& src) {
        deepcopy<SrcSpace, T_src>(&host_constant(), src);
        deepcopy<DestSpace, T_dst, T_src>(&device_constant(), src);
    }

    // For template types
    template <typename U = T<CurrentSpace>,
              std::enable_if_t<is_space_template_type_v<U>, bool> = true> 
    static __host__ __device__ U& get() {
        #ifdef __CUDA_ARCH__
            return device_constant();
        #else
            return host_constant();
        #endif
    }
};
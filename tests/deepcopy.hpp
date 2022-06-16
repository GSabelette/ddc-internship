#include <Kokkos_Core.hpp>
#include <type_traits>
#include <stdio.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using DefaultSpace = Kokkos::HostSpace;

template <typename T>
struct is_default_handled_type : std::integral_constant<bool, (std::is_arithmetic_v<T> || 
    std::is_pointer_v<T> || Kokkos::is_view<T>::value)> {};

template <typename T>
inline constexpr bool is_default_handled_type_v = is_default_handled_type<T>::value;

// template <typename T>
// struct is_not_default_handled_type : std::integral_constant<bool, !(std::is_arithmetic<T>::value || 
//     std::is_pointer<T>::value || Kokkos::is_view<T>::value)> {};

// template <typename T>
// inline constexpr bool is_not_default_handled_type_v = is_not_default_handled_type<T>::value;

template <class DestSpace, typename T>
T* empty_alloc() {
    return static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T)));
}

template <class DestSpace, typename T>
T* empty_alloc(const std::size_t n) {
    return static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T) * n));
}

template <typename T_dst, typename T_src = T_dst>
void shallow_copy(T_dst* dst, const T_src& src) {
    cudaMemcpy(dst, &src, sizeof(T_src), cudaMemcpyDefault);
    cudaCheckErrors("deepcopy");
}

template <typename T_dst, typename T_src = T_dst>
void shallow_copy(T_dst* dst, const T_src& src, const std::size_t n) {
    cudaMemcpy(dst, &src, sizeof(T_src) * n, cudaMemcpyDefault);
    cudaCheckErrors("deepcopy");
}

// Arithmetic types.
template <class DestSpace, typename T_dst, typename T_src = T_dst, 
          std::enable_if_t<std::is_arithmetic<T_dst>::value, bool> = true>
void deepcopy(T_dst* dst, const T_src& src) {
    shallow_copy<T_dst, T_src>(dst, src);
}

// Pointer types.
template <class DestSpace, typename T_dst, typename T_src = T_dst, 
          std::enable_if_t<std::is_pointer<T_dst>::value, bool> = true>
void deepcopy(T_dst dst, const T_src& src) {
    using T_src_plain = std::remove_pointer_t<T_src>;
    using T_dst_plain = std::remove_pointer_t<T_dst>;
    
    *dst = empty_alloc<DestSpace, T_dst_plain>();
    deepcopy<DestSpace, T_dst_plain, T_src_plain>(*dst, *src);
}

// Array types.
template <class DestSpace, typename T_dst, typename T_src = T_dst, 
          std::enable_if_t<std::is_pointer<T_dst>::value, bool> = true>
void deepcopy(T_dst* dst, const T_src& src, const std::size_t n) {
    using T_src_plain = std::remove_pointer_t<T_src>;
    using T_dst_plain = std::remove_pointer_t<T_dst>;

    *dst = empty_alloc<DestSpace, T_dst_plain>(n);
    shallow_copy<T_dst_plain, T_src_plain>(*dst, *src, n);
}

// View types.
template <class DestSpace, typename T_dst, typename T_src = T_dst, 
          std::enable_if_t<Kokkos::is_view<T_dst>::value, bool> = true>
void deepcopy(T_dst* dst, const T_src& src) {
    Kokkos::deep_copy(*dst, src);
}

// User types. 
template <class DestSpace, typename T_dst, typename T_src = T_dst,
          std::enable_if_t<!is_default_handled_type_v<T_dst>, bool> = true>
void deepcopy(T_dst* dst, const T_src& src) {
    T_dst tmp(src);
    shallow_copy<T_dst>(dst, tmp);
}

// Cloning.
template <class DestSpace, typename T_dst, typename T_src = T_dst> 
T_dst* clone(const T_src& src) {
    T_dst* dst = empty_alloc<DestSpace, T_dst>();
    deepcopy<DestSpace, T_dst, T_src>(dst, src);
    return dst;
}
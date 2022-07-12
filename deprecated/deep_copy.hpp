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

// Default space for clone methods.
using DefaultSpace = Kokkos::HostSpace;

// stolen from user bit2shift on stackoverflow.
template<typename, typename>
constexpr bool is_same_template{false};

template<
    template<typename...> typename T,
    typename... A,
    typename... B
>
constexpr bool is_same_template<
    T<A...>,
    T<B...>
>{true};

template <typename T, class DestSpace>
T* empty_alloc() {
    return static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T)));
}

template <template <class> typename T, class DestSpace>
T<DestSpace>* empty_alloc_destspace() {
    return static_cast<T<DestSpace>*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T<DestSpace>)));
}

template <typename T>
void deepcopy(T* dst, const T& src) {
    cudaMemcpy(dst, &src, sizeof(T), cudaMemcpyDefault);
    cudaCheckErrors("deepcopy");
}

template<template <class> typename T, class DestSpace, class OriginSpace>
void deepcopy(T<DestSpace>* dst, const T<OriginSpace>& src) {
    if (Kokkos::is_view<T<void>>::value) {
        Kokkos::deep_copy(*dst, src);
    }
    else {
        printf("Non view is not supported yet");
        return;
    }
}

// allocates memory pointed to by dst and copies memory pointed by src into it.
template <typename T, class DestSpace>
void deepcopyptr(T** dst, const T* src) {
    T* tmp = empty_alloc<T, DestSpace>();
    deepcopy<T>(tmp, *src);
    deepcopy<T*>(dst, tmp);
}

// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4502.pdf.
template <typename...>
using void_t = void;

// Primary template handles all types not supporting the operation.
template <typename, template <typename> class, typename = void_t<>>
struct detect : std::false_type {};

// Specialization recognizes/validates only types supporting the archetype.
template <typename T, template <typename> class Op>
struct detect<T, Op, void_t<Op<T>>> : std::true_type {};

template <typename T>
using clone_t = decltype(std::declval<T>().clone());

template <typename T>
using has_clone = detect<T, clone_t>;

template <typename T>
constexpr bool has_clone_v = detect<T, clone_t>::value;

template <typename T, class DestSpace> 
void member_copy(T* dst, const T& src) {  
    if constexpr (has_clone_v<T>){
        printf("member copy : clone\n");
        deepcopy<T>(dst, src.template clone<DestSpace>());
    }
    else if constexpr (std::is_pointer_v<T>) {
        if constexpr (has_clone_v<std::remove_pointer_t<T>>) {
            printf("member copy : non-trivial deepcopyptr\n");
            T tmp = src->template clone<DestSpace>();
            deepcopy<T>(dst, tmp);
        }
        else if constexpr (std::is_trivially_copyable_v<std::remove_pointer_t<T>>) {
            printf("member copy : trivial deepcopyptr\n");
            //printf("from adress : %p\n", src);
            printf("targeted adress : %p\n", dst);
            deepcopyptr<std::remove_pointer_t<T>, DestSpace>(dst, src);
        }
        else printf("Non trivially copyable pointer and no clone method was provided\n");
    }
    else if constexpr (std::is_trivially_copyable_v<T>) {
        printf("member copy : trivial || trivially copyable deepcopy\n");
        deepcopy<T>(dst, src);
    }
    else printf("Non trivially copyable and no clone method was provided\n");
}
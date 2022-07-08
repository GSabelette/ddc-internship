#include <Kokkos_Core.hpp>
#include <type_traits>
#include <stdio.h>
#include <vector>
#include <algorithm>

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

template<typename T> struct remove_all {
    typedef T type;
};
template<typename T> struct remove_all<T*> {
    typedef typename remove_all<T>::type type;
};

using DefaultSpace = Kokkos::HostSpace;

template <typename T>
struct is_default_handled_type : std::integral_constant<bool, (std::is_arithmetic_v<T> || 
    std::is_pointer_v<T> || Kokkos::is_view<T>::value)> {};

template <typename T>
inline constexpr bool is_default_handled_type_v = is_default_handled_type<T>::value;

template <typename ViewType>
std::size_t get_size(const ViewType& v) {
    std::size_t size = sizeof(typename ViewType::value_type);
    for (int i = 0; i < ViewType::rank; size *= v.extent(i), ++i);
    return size;
}


namespace Deepcopy {
    std::vector<void*> host_queue {};
    std::vector<void*> device_queue {};

    void print_queues() {
        printf("host_queue : ");
        for (auto& p : host_queue) printf("%p, ", p);
        printf("\ndevice_queue : ");
        for (auto& p : device_queue) printf("%p, ", p);
        printf("\n");
    }

    template <class MemorySpace,
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::HostSpace>, bool> = true>
    void add_to_queue(void* ptr) {
        host_queue.push_back(ptr);
    }

    template <class MemorySpace,
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::CudaSpace>, bool> = true>
    void add_to_queue(void* ptr) {
        device_queue.push_back(ptr);
    }

    template <class MemorySpace, 
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::HostSpace>, bool> = true>
    void free(void* ptr) {
        std::vector<void*>::iterator pos = std::find(host_queue.begin(), host_queue.end(), ptr);
        if (pos != host_queue.end()) {
            host_queue.erase(pos);
            printf("freed on CPU : %p\n", ptr);
            Kokkos::kokkos_free<MemorySpace>(ptr);   
        }
    }

    template <class MemorySpace, 
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::CudaSpace>, bool> = true>
    void free(void* ptr) {
        std::vector<void*>::iterator pos = std::find(device_queue.begin(), device_queue.end(), ptr);
        if (pos != device_queue.end()) {
            device_queue.erase(pos);
            printf("freed on GPU : %p\n", ptr);
            Kokkos::kokkos_free<MemorySpace>(ptr);   
        }
    }

    void free(void* ptr) {
        std::vector<void*>::iterator pos = std::find(host_queue.begin(), host_queue.end(), ptr);
        if (pos != host_queue.end()) {
            host_queue.erase(pos);
            printf("freed on CPU : %p\n", ptr);
            Kokkos::kokkos_free<Kokkos::HostSpace>(ptr);   
        } else {
            std::vector<void*>::iterator pos = std::find(device_queue.begin(), device_queue.end(), ptr);
            if (pos != device_queue.end()) {
                device_queue.erase(pos);
                printf("freed on GPU : %p\n", ptr);
                Kokkos::kokkos_free<Kokkos::CudaSpace>(ptr);   
            }
        }
    }

    void clear() {
        std::for_each(host_queue.begin(), host_queue.end(), Kokkos::kokkos_free<Kokkos::HostSpace>);
        std::for_each(device_queue.begin(), device_queue.end(), Kokkos::kokkos_free<Kokkos::CudaSpace>);
        host_queue.clear();
        device_queue.clear();
    }
};

template <typename data_type>
using DeepcopyableViewType = Kokkos::View<data_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename data_type, class DestSpace>
using DeepcopyableView = Kokkos::View<data_type, DestSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class DestSpace, typename ViewType>
typename ViewType::value_type* deepcopyable_view_alloc(ViewType src_view) {
    typename ViewType::value_type* ptr = static_cast<typename ViewType::value_type*>(Kokkos::kokkos_malloc<DestSpace>(get_size(src_view)));
    Deepcopy::add_to_queue<DestSpace>(ptr);
    return ptr;
}

template <class DestSpace, typename data_type>
typename remove_all<data_type>::type* deepcopyable_view_alloc(std::size_t size) {
    using ptr_type = typename remove_all<data_type>::type*;
    ptr_type ptr = static_cast<ptr_type>(Kokkos::kokkos_malloc<DestSpace>(size));
    Deepcopy::add_to_queue<DestSpace>(ptr);
    return ptr;
}

template <typename data_type, class DestSpace = DefaultSpace, typename... Args>
DeepcopyableView<data_type, DestSpace> deepcopyable_view(std::size_t size, Args... args) {
    DeepcopyableView<data_type, DestSpace> v(deepcopyable_view_alloc<DestSpace, data_type>(size), args...);
    return v;
}

// Fill methods to initialize arrays/views independently of memory space
template <typename Lambda, typename ViewType, 
          std::enable_if_t<std::is_same_v<typename ViewType::memory_space, Kokkos::HostSpace>, bool> = true>
__host__ __device__ void fill_view(ViewType& v, Lambda&& f) {
    f(v);
}

template <typename Lambda, typename ViewType,
          std::enable_if_t<std::is_same_v<typename ViewType::memory_space, Kokkos::CudaSpace>, bool> = true>
__host__ __device__ void fill_view(ViewType& v, Lambda&& f) {
    typename ViewType::HostMirror h_v;
    h_v = Kokkos::create_mirror_view(v);
    f(h_v);
    Kokkos::deep_copy(v, h_v); 
}

template <class DestSpace, typename Lambda, typename T,
          std::enable_if_t<std::is_same_v<DestSpace, Kokkos::HostSpace>, bool> = true>
void fill_array(T* ptr, Lambda&& f, std::size_t array_size) {
    f(ptr);
}

template <class DestSpace, typename Lambda, typename T,
          std::enable_if_t<std::is_same_v<DestSpace, Kokkos::CudaSpace>, bool> = true>
void fill_array(T* ptr, Lambda&& f, std::size_t array_size) {
    T* tmp = new T;
    f(tmp);
    cudaMemcpy(ptr, tmp, array_size, cudaMemcpyHostToDevice);
}

template <class DestSpace, typename T>
T* empty_alloc() {
    T* ptr = static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T)));
    Deepcopy::add_to_queue<DestSpace>(ptr);
    return ptr;
}

template <class DestSpace, typename T>
T* empty_alloc(const std::size_t n) {
    T* ptr = static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T) * n));
    Deepcopy::add_to_queue<DestSpace>(ptr);
    return ptr;
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

// Array of simple types.
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
template <class DestSpace, typename T_dst, typename T_src,
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
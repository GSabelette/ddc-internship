#include <Kokkos_Core.hpp>
#include <type_traits>
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>
#include <new>

#if defined (__CUDA_ARCH__)
using CurrentSpace = Kokkos::CudaSpace;
#else
using CurrentSpace = Kokkos::HostSpace;
#endif

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
    std::vector<void*> parent_stack {};
    std::vector<void*> roots {}; 
    std::map<void*, std::vector<void*>> tree;
    std::vector<void*> device_nodes {};

    std::vector<void*> host_queue {};
    std::vector<void*> device_queue {};

    void print_queues() {
        printf("host_queue : ");
        for (auto& p : host_queue) printf("%p, ", p);
        printf("\ndevice_queue : ");
        for (auto& p : device_queue) printf("%p, ", p);
        printf("\n");
    }

    void print_tree() {
        printf("Tree : ");
        for (const auto& [key, value] : tree) {
            printf("parent : %p | sons : {", key);
            for (auto& v : value) printf("%p ,", v);
            printf("} || ");
        }
        printf("\n");
    }

    void print_roots() {
        printf("roots : ");
        for (auto& p : roots) printf("%p, ", p);
        printf("\n");
    }

    void print_parent_stack() {
        printf("parent_stack : ");
        for (auto& p : parent_stack) printf("%p, ", p);
        printf("\n");        
    }

    template <class MemorySpace,
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::HostSpace>, bool> = true>
    void add_to_tree(void* ptr) {
        printf("adding %p to tree\n", ptr);
        tree[ptr] = {};
        if (!parent_stack.empty()) {
            printf("pushed %p to parent %p\n", ptr, parent_stack.back());
            tree[parent_stack.back()].push_back(ptr);
        } else {
            roots.push_back(ptr);
        }
    }

    template <class MemorySpace,
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::CudaSpace>, bool> = true>
    void add_to_tree(void* ptr) {
        printf("adding %p to tree\n", ptr);
        tree[ptr] = {};
        if (!parent_stack.empty()) {
            printf("pushed to parent\n");
            tree[parent_stack.back()].push_back(ptr);
        } else {
            roots.push_back(ptr);
        }
        device_nodes.push_back(ptr);
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

    void tree_free(void* ptr) {
        if (!tree[ptr].empty()) {
            for (void* children : tree[ptr]) {
                tree_free(children);
             }
        }
        tree.erase(ptr);
        if (std::find(roots.begin(), roots.end(), ptr) != roots.end()) roots.erase(std::find(roots.begin(), roots.end(), ptr)); 
        if (std::find(device_nodes.begin(), device_nodes.end(), ptr) == device_nodes.end()) Kokkos::kokkos_free<Kokkos::HostSpace>(ptr); 
        else Kokkos::kokkos_free<Kokkos::CudaSpace>(ptr);
    }

    void tree_clear() {
        for (auto& root : roots) {
            tree_free(root);
        }
    }

    template <class MemorySpace, 
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::HostSpace>, bool> = true>
    void free(void* ptr) {
        std::vector<void*>::iterator pos = std::find(host_queue.begin(), host_queue.end(), ptr);
        if (pos != host_queue.end()) {
            host_queue.erase(pos);
            Kokkos::kokkos_free<MemorySpace>(ptr);   
        }
    }

    template <class MemorySpace, 
              std::enable_if_t<std::is_same_v<MemorySpace, Kokkos::CudaSpace>, bool> = true>
    void free(void* ptr) {
        std::vector<void*>::iterator pos = std::find(device_queue.begin(), device_queue.end(), ptr);
        if (pos != device_queue.end()) {
            device_queue.erase(pos);
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


    template <typename data_type, class DestSpace>
    using View = Kokkos::View<data_type, DestSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template <class DestSpace, typename ViewType>
    typename ViewType::value_type* view_alloc(ViewType src_view) {
        typename ViewType::value_type* ptr = static_cast<typename ViewType::value_type*>(Kokkos::kokkos_malloc<DestSpace>(get_size(src_view)));
        add_to_tree<DestSpace>(ptr);
        return ptr;
    }

    template <class DestSpace, typename data_type>
    typename remove_all<data_type>::type* view_alloc(std::size_t size) {
        using ptr_type = typename remove_all<data_type>::type*;
        ptr_type ptr = static_cast<ptr_type>(Kokkos::kokkos_malloc<DestSpace>(size));
        add_to_tree<DestSpace>(ptr);
        return ptr;
    }

    template <typename data_type, class DestSpace = DefaultSpace, typename... Args>
    View<data_type, DestSpace> view(std::size_t size, Args... args) {
        View<data_type, DestSpace> v(view_alloc<DestSpace, data_type>(size), args...);
        return v;
    }
};

template <class DestSpace, typename T>
T* empty_alloc() {
    T* ptr = static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T)));
    Deepcopy::add_to_tree<DestSpace>(ptr);
    return ptr;
}

template <class DestSpace, typename T>
T* empty_alloc(const std::size_t n) {
    T* ptr = static_cast<T*>(Kokkos::kokkos_malloc<DestSpace>(sizeof(T) * n));
    Deepcopy::add_to_tree<DestSpace>(ptr);
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

// View types.
template <class DestSpace, typename T_dst, typename T_src = T_dst, 
          std::enable_if_t<Kokkos::is_view<T_dst>::value, bool> = true>
void deepcopy(T_dst* dst, const T_src& src) {
    Kokkos::deep_copy(*dst, src);
}

// TODO : implement View-types deepcopy that also calls the constructor and allocates memory.

// User types. 
template <class DestSpace, typename T_dst, typename T_src,
          std::enable_if_t<!is_default_handled_type_v<T_dst>, bool> = true>
void deepcopy(T_dst* dst, const T_src& src) {
    Deepcopy::parent_stack.push_back(&dst);
    T_dst tmp(src);
    Deepcopy::parent_stack.pop_back();
    shallow_copy<T_dst>(dst, tmp);
}

// Pointer types.
template <class DestSpace, typename T_dst, typename T_src = T_dst, 
          std::enable_if_t<std::is_pointer<T_dst>::value, bool> = true>
void deepcopy(T_dst* dst, const T_src& src) {
    using T_src_plain = std::remove_pointer_t<T_src>;
    using T_dst_plain = std::remove_pointer_t<T_dst>;
    
    *dst = empty_alloc<DestSpace, T_dst_plain>();
    Deepcopy::parent_stack.push_back(&dst);
    deepcopy<DestSpace, T_dst_plain, T_src_plain>(*dst, *src);
    Deepcopy::parent_stack.pop_back();
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

// Cloning.
template <class DestSpace, typename T_dst, typename T_src> 
T_dst* clone(const T_src& src) {
    T_dst* dst = empty_alloc<DestSpace, T_dst>();
    deepcopy<DestSpace, T_dst, T_src>(dst, src);
    return dst;
}


// Fill methods to initialize arrays/views independently of memory space
// CPU to CPU.
template <typename Lambda, typename ViewType, 
          std::enable_if_t<std::is_same_v<typename ViewType::memory_space, Kokkos::HostSpace>
          && std::is_same_v<CurrentSpace, Kokkos::HostSpace>, bool> = true>
void fill_view(ViewType& v, Lambda&& f) {
    f(v);
}

//GPU to CPU. Just used as a check so CPU to CPU is not misscalled.
template <typename Lambda, typename ViewType, 
          std::enable_if_t<std::is_same_v<typename ViewType::memory_space, Kokkos::HostSpace>
          && std::is_same_v<CurrentSpace, Kokkos::CudaSpace>, bool> = true>
__device__ void fill_view(ViewType& v, Lambda&& f) {
    return;
}

// CPU to GPU.
template <typename Lambda, typename ViewType,
          std::enable_if_t<std::is_same_v<typename ViewType::memory_space, Kokkos::CudaSpace>
          && std::is_same_v<CurrentSpace, Kokkos::HostSpace>, bool> = true>
void fill_view(ViewType& v, Lambda&& f) {
    typename ViewType::HostMirror h_v;
    h_v = Kokkos::create_mirror_view(v);
    f(h_v);
    Kokkos::deep_copy(v, h_v); 
}

// GPU to GPU. Just used as a check so CPU to GPU is not misscalled.
template <typename Lambda, typename ViewType,
          std::enable_if_t<std::is_same_v<typename ViewType::memory_space, Kokkos::CudaSpace>
          && std::is_same_v<CurrentSpace, Kokkos::CudaSpace>, bool> = true>
__device__ void fill_view(ViewType& v, Lambda&& f) {
    return;
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
    cudaMemcpy(ptr, tmp, array_size, cudaMemcpyDefault);
}

template <class DestSpace, typename T, typename... Args,
          std::enable_if_t<std::is_same_v<DestSpace, Kokkos::HostSpace>, bool> = true>
void init_alloc(T** ptr, Args... args) {
    *ptr = empty_alloc<DestSpace, T>();
    Deepcopy::parent_stack.push_back(*ptr);
    *ptr = new (*ptr) T(args...);
    Deepcopy::parent_stack.pop_back();
}

template <class DestSpace, typename T, typename... Args,
          std::enable_if_t<std::is_same_v<DestSpace, Kokkos::CudaSpace>, bool> = true>
void init_alloc(T** ptr, Args... args) {
    // Small trick to be able to free memory allocated by our temp.
    T* tmp = empty_alloc<DefaultSpace, T>();
    Deepcopy::add_to_tree<DefaultSpace>(tmp);

    Deepcopy::parent_stack.push_back(tmp);
    tmp = new T(args...);
    Deepcopy::parent_stack.pop_back();
    
    // printf("tmp : %p\n", tmp);
    // Deepcopy::print_tree();

    // pointer-type deepcopy handles allocation, copy and addition to the reference tree.
    deepcopy<DestSpace, T*>(ptr, tmp);
    //Deepcopy::tree_free(tmp);
}
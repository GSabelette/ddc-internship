#include <iostream>
#include <stdio.h>
#include <type_traits>

#if defined (__CUDA_ARCH__)
using foo = int;
#else 
using foo = char*;
#endif

template <typename return_type>
__host__ __device__ return_type bar() {
    if constexpr (std::is_same<return_type, int>::value) return 256;
    else if constexpr (std::is_same<return_type, char*>::value) return "bar";
    else return nullptr;
}

__host__ __device__ auto baz() {
    if constexpr(std::is_same_v<foo, int>) return 256;
    else if constexpr(std::is_same_v<foo, char*>) return "doot";
    // #if defined (__CUDA_ARCH__) 
    // return 256;
    // #else 
    // return "doot";
    // #endif
}

template <typename T = foo>
__host__ __device__ void boop() {
    printf("is_int : %d\n", std::is_same<T, int>::value);
    printf("is_char* : %d\n", std::is_same<T, char*>::value);
    T bat;
    if constexpr (std::is_same<T, int>::value) {
        bat = 256;
        printf("Kernel boop() : %d\n", bat);
    }
    else if constexpr (std::is_same<T, char*>::value) {
        bat = "boop";
        printf("CPU boop() : %s\n", bat);
    }
    else {
        printf("type was not recognized\n");
    }
}

__global__ void kernel() {
    foo test = baz();
    printf("Kernel bar() : %d\n", bar<foo>());
    printf("Kernel baz() : %d\n", baz());
    printf("Kernel baz() : %d\n", test);

    boop();
}

int main() {
    std::cout << "Host bar() : " << bar<foo>() << "\n";
    std::cout << "Host baz() : " << baz() << "\n";
    std::cout << "Host boop() : "; boop();
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}
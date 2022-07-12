cmake_minimum_required(VERSION 3.15)
project(benchmark_game_of_life CXX)

option(CUDA_DEVICE "" ON)

add_subdirectory(vendor/kokkos)
find_package(benchmark)
if (NOT benchmark_FOUND)
    add_subdirectory(vendor/benchmark)
endif()
add_executable(benchmark_game_of_life benchmark_game_of_life.cpp)
target_link_libraries(benchmark_game_of_life
    PUBLIC
        benchmark::benchmark)
target_link_libraries(benchmark_game_of_life
    PUBLIC
        Kokkos::kokkos)

if ("${CUDA_DEVICE}")
    target_compile_definitions(benchmark_game_of_life PUBLIC -DCUDA_DEVICE=1)
    add_executable(validity_test validity_test.cpp)
    target_link_libraries(validity_test
    PUBLIC
        Kokkos::kokkos)
else ()
    target_compile_definitions(benchmark_game_of_life PUBLIC -DCUDA_DEVICE=0)
endif()

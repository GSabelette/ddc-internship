cmake_minimum_required(VERSION 3.15)
project(benchmark_game_of_life CXX)

option(CUDA_DEVICE "" OFF)

add_subdirectory(vendor/kokkos)
find_package(benchmark)
if (NOT benchmark_FOUND)
    add_subdirectory(vendor/benchmark)
endif()
add_executable(benchmark_game_of_life benchmark_game_of_life.cpp)
#add_executable(deepcopy_test deepcopy_test.cpp)
# add_executable(global_view_test global_view_test.cpp)
target_link_libraries(benchmark_game_of_life
    PUBLIC
        benchmark::benchmark)
target_link_libraries(benchmark_game_of_life
    PUBLIC
        Kokkos::kokkos)
# target_link_libraries(deepcopy_test
#         PUBLIC
#             Kokkos::kokkos)
# target_link_libraries(global_view_test
#         PUBLIC
#             Kokkos::kokkos)

#set_property(TARGET transparent_global_test PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)

if ("${CUDA_DEVICE}")
    target_compile_definitions(benchmark_game_of_life PUBLIC -DCUDA_DEVICE=1)
    add_executable(validity_test validity_test.cpp)
    target_link_libraries(validity_test
    PUBLIC
        Kokkos::kokkos)
else ()
    target_compile_definitions(benchmark_game_of_life PUBLIC -DCUDA_DEVICE=0)
endif()

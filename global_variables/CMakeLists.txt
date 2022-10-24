cmake_minimum_required(VERSION 3.15)
project(global_variables CXX)

add_subdirectory(vendor/kokkos)
add_executable(unmanaged_view_test unmanaged_view_test.cpp)
add_executable(global_var_test global_var_test.cpp)
add_executable(tree_test tree_test.cpp)
target_link_libraries(unmanaged_view_test
            PUBLIC
                Kokkos::kokkos)
target_link_libraries(global_var_test
        PUBLIC
            Kokkos::kokkos)
target_link_libraries(tree_test
        PUBLIC
            Kokkos::kokkos)
cmake_minimum_required(VERSION 3.15)
project(tests CXX)

add_subdirectory(vendor/kokkos)
add_executable(deepcopy_test deepcopy_test.cpp)
add_executable(unmanaged_view_test unmanaged_view_test.cpp)
# add_executable(global_var_test global_var_test.cpp)
#add_executable(placement_new_test placement_new_test.cpp)
target_link_libraries(deepcopy_test
        PUBLIC
            Kokkos::kokkos)
target_link_libraries(unmanaged_view_test
            PUBLIC
                Kokkos::kokkos)
# target_link_libraries(global_var_test
#         PUBLIC
#             Kokkos::kokkos)
# target_link_libraries(placement_new_test
#         PUBLIC
#             Kokkos::kokkos)

#set_property(TARGET transparent_global_test PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)
# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-11.2.0/cmake-3.21.4-7ggmda7wvxddqzhvi5goo7pbwvoy7u4e/bin/cmake

# The command to remove a file.
RM = /gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-11.2.0/cmake-3.21.4-7ggmda7wvxddqzhvi5goo7pbwvoy7u4e/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /gpfs/users/anquetilm/ddc_internship/tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gpfs/users/anquetilm/ddc_internship/tests/build

# Include any dependencies generated for this target.
include CMakeFiles/global_var_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/global_var_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/global_var_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/global_var_test.dir/flags.make

CMakeFiles/global_var_test.dir/global_var_test.cpp.o: CMakeFiles/global_var_test.dir/flags.make
CMakeFiles/global_var_test.dir/global_var_test.cpp.o: ../global_var_test.cpp
CMakeFiles/global_var_test.dir/global_var_test.cpp.o: CMakeFiles/global_var_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gpfs/users/anquetilm/ddc_internship/tests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/global_var_test.dir/global_var_test.cpp.o"
	/gpfs/users/anquetilm/ddc_internship/tests/vendor/kokkos/bin/kokkos_launch_compiler /gpfs/users/anquetilm/ddc_internship/tests/vendor/kokkos/bin/nvcc_wrapper /gpfs/softs/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/gcc-11.2.0-mpv3i3uebzvnvz7wxn6ywysd5hftycj3/bin/g++ /gpfs/softs/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/gcc-11.2.0-mpv3i3uebzvnvz7wxn6ywysd5hftycj3/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/global_var_test.dir/global_var_test.cpp.o -MF CMakeFiles/global_var_test.dir/global_var_test.cpp.o.d -o CMakeFiles/global_var_test.dir/global_var_test.cpp.o -c /gpfs/users/anquetilm/ddc_internship/tests/global_var_test.cpp

CMakeFiles/global_var_test.dir/global_var_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/global_var_test.dir/global_var_test.cpp.i"
	/gpfs/softs/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/gcc-11.2.0-mpv3i3uebzvnvz7wxn6ywysd5hftycj3/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gpfs/users/anquetilm/ddc_internship/tests/global_var_test.cpp > CMakeFiles/global_var_test.dir/global_var_test.cpp.i

CMakeFiles/global_var_test.dir/global_var_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/global_var_test.dir/global_var_test.cpp.s"
	/gpfs/softs/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/gcc-11.2.0-mpv3i3uebzvnvz7wxn6ywysd5hftycj3/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gpfs/users/anquetilm/ddc_internship/tests/global_var_test.cpp -o CMakeFiles/global_var_test.dir/global_var_test.cpp.s

# Object files for target global_var_test
global_var_test_OBJECTS = \
"CMakeFiles/global_var_test.dir/global_var_test.cpp.o"

# External object files for target global_var_test
global_var_test_EXTERNAL_OBJECTS =

global_var_test: CMakeFiles/global_var_test.dir/global_var_test.cpp.o
global_var_test: CMakeFiles/global_var_test.dir/build.make
global_var_test: vendor/kokkos/containers/src/libkokkoscontainers.a
global_var_test: vendor/kokkos/core/src/libkokkoscore.a
global_var_test: /usr/lib64/libcuda.so
global_var_test: /gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-11.2.0/cuda-11.5.0-kij363kedhc4g7hbpxakhang3ypoxnst/lib64/libcudart.so
global_var_test: /usr/lib64/libdl.so
global_var_test: CMakeFiles/global_var_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/gpfs/users/anquetilm/ddc_internship/tests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable global_var_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/global_var_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/global_var_test.dir/build: global_var_test
.PHONY : CMakeFiles/global_var_test.dir/build

CMakeFiles/global_var_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/global_var_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/global_var_test.dir/clean

CMakeFiles/global_var_test.dir/depend:
	cd /gpfs/users/anquetilm/ddc_internship/tests/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfs/users/anquetilm/ddc_internship/tests /gpfs/users/anquetilm/ddc_internship/tests /gpfs/users/anquetilm/ddc_internship/tests/build /gpfs/users/anquetilm/ddc_internship/tests/build /gpfs/users/anquetilm/ddc_internship/tests/build/CMakeFiles/global_var_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/global_var_test.dir/depend


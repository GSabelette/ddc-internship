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

# Utility rule file for ExperimentalBuild.

# Include any custom commands dependencies for this target.
include vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/compiler_depend.make

# Include the progress variables for this target.
include vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/progress.make

vendor/kokkos/CMakeFiles/ExperimentalBuild:
	cd /gpfs/users/anquetilm/ddc_internship/tests/build/vendor/kokkos && /gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-11.2.0/cmake-3.21.4-7ggmda7wvxddqzhvi5goo7pbwvoy7u4e/bin/ctest -D ExperimentalBuild

ExperimentalBuild: vendor/kokkos/CMakeFiles/ExperimentalBuild
ExperimentalBuild: vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/build.make
.PHONY : ExperimentalBuild

# Rule to build all files generated by this target.
vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/build: ExperimentalBuild
.PHONY : vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/build

vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/clean:
	cd /gpfs/users/anquetilm/ddc_internship/tests/build/vendor/kokkos && $(CMAKE_COMMAND) -P CMakeFiles/ExperimentalBuild.dir/cmake_clean.cmake
.PHONY : vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/clean

vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/depend:
	cd /gpfs/users/anquetilm/ddc_internship/tests/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfs/users/anquetilm/ddc_internship/tests /gpfs/users/anquetilm/ddc_internship/tests/vendor/kokkos /gpfs/users/anquetilm/ddc_internship/tests/build /gpfs/users/anquetilm/ddc_internship/tests/build/vendor/kokkos /gpfs/users/anquetilm/ddc_internship/tests/build/vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vendor/kokkos/CMakeFiles/ExperimentalBuild.dir/depend


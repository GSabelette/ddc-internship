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
CMAKE_SOURCE_DIR = /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build

# Utility rule file for Nightly.

# Include any custom commands dependencies for this target.
include vendor/kokkos/CMakeFiles/Nightly.dir/compiler_depend.make

# Include the progress variables for this target.
include vendor/kokkos/CMakeFiles/Nightly.dir/progress.make

vendor/kokkos/CMakeFiles/Nightly:
	cd /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build/vendor/kokkos && /gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-11.2.0/cmake-3.21.4-7ggmda7wvxddqzhvi5goo7pbwvoy7u4e/bin/ctest -D Nightly

Nightly: vendor/kokkos/CMakeFiles/Nightly
Nightly: vendor/kokkos/CMakeFiles/Nightly.dir/build.make
.PHONY : Nightly

# Rule to build all files generated by this target.
vendor/kokkos/CMakeFiles/Nightly.dir/build: Nightly
.PHONY : vendor/kokkos/CMakeFiles/Nightly.dir/build

vendor/kokkos/CMakeFiles/Nightly.dir/clean:
	cd /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build/vendor/kokkos && $(CMAKE_COMMAND) -P CMakeFiles/Nightly.dir/cmake_clean.cmake
.PHONY : vendor/kokkos/CMakeFiles/Nightly.dir/clean

vendor/kokkos/CMakeFiles/Nightly.dir/depend:
	cd /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/vendor/kokkos /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build/vendor/kokkos /gpfs/users/anquetilm/kokkos_game_of_life/kokkos_game_of_life/build/vendor/kokkos/CMakeFiles/Nightly.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vendor/kokkos/CMakeFiles/Nightly.dir/depend


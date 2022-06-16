#!/bin/bash

set -e

module purge

module load gcc/11.2.0/gcc-4.8.5
module load cmake/3.21.4/gcc-11.2.0
module load cuda/11.5.0/gcc-11.2.0

module list > ../logs/openmp_modules.log

env &> ../logs/openmp_env.log


cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_CXX_STANDARD=17 \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ARCH_SKX=ON .. &> ../logs/openmp_compil.log
make VERBOSE=1 -j 8 &>> ../logs/openmp_compil.log

sbatch ../sbatch_openmp.sh
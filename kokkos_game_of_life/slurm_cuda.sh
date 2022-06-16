#!/bin/bash
#SBATCH --job-name=slurm_cuda
#SBATCH --output=../slurm_out/gpu.out
#SBATCH --error=../err/gpu.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e 

module purge
module load gcc/11.2.0/gcc-4.8.5
module load cuda/11.5.0/gcc-11.2.0
module load cmake/3.16.2/gcc-9.2.0

module list > ../logs/cuda_modules.log

env &> ../logs/cuda_env.log

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_CXX_STANDARD=17 \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_VOLTA70=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_DEPRECATED_CODE_3=OFF \
  -DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  .. &> ../logs/cuda_compil.log
make VERBOSE=1 -j 8 &>> ../logs/cuda_compil.log

set -x

echo $SLURM_STEP_GPUS

./validity_test  > ../logs/validity_test.log

./benchmark_game_of_life --benchmark_out=../out/gpu.json  --benchmark_out_format=json
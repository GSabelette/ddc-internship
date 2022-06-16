#!/bin/bash
#SBATCH --job-name=cudatest
#SBATCH --output=/out/cudatest.out
#SBATCH --error=/err/cudatest.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e 

module purge
module load gcc/11.2.0/gcc-4.8.5
module load cuda/11.5.0/gcc-11.2.0

nvcc -v -gencode arch=compute_70,code=sm_70 ../transparent_global_test.cu -o transparent_global_test &> ../logs/transparent_global_test_compil.log

echo $SLURM_STEP_GPUS

set -x

./test &> ./logs/transparent_global_test.log

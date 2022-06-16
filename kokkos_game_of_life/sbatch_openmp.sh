#!/bin/bash
#SBATCH --job-name=slurm_openmp
#SBATCH --time=01:00:00
#SBATCH --array=1-40%5
#SBATCH --output=../slurm_out/openmp.out
#SBATCH --error=../err/%acpu.err
#SBATCH --mem=4G
#SBATCH --partition=cpu_med

set -e

SLURM_CPUS_PER_TASK=${SLURM_ARRAY_TASK_ID}

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# number of OpenMP threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} 

# Binding OpenMP threads on core
export OMP_PLACES=cores

# execution with 'OMP_NUM_THREADS' OpenMP threads
./benchmark_game_of_life --benchmark_out=../out/${SLURM_ARRAY_TASK_ID}cpu.json  --benchmark_out_format=json

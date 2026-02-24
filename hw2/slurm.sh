#!/bin/bash
SBATCH --job-name=pingpong_run
SBATCH --output=%x_%j.out
SBATCH --error=%x_%j.err
SBATCH --time=00:05:00
SBATCH --export=ALL


### Uncomment these lines if you want to hardcode the number of nodes and number of processes (same as tasks) inside this job script
### commented #SBATCH --nodes=4
### commented #SBATCH --ntasks=16

####### uncomment these lines if you haven't loaded the modules already
##### module load gcc
##### module load mpich


echo "Environment on $(hostname)"
echo "Running MPI Hello..."
srun ./pingpong



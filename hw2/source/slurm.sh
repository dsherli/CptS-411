#!/bin/bash
#SBATCH --job-name=allreduce
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=32
#SBATCH --export=ALL

module load gcc
module load mpich

set -e

REPS=5

for LOG_P in 0 1 2 3 4 5; do        # p = 1,2,4,8,16,32
  P=$((1 << LOG_P))
  srun -n $P ./reduce $REPS
done

echo "All done."

#!/bin/bash
#SBATCH --job-name=allreduce
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=32
#SBATCH --export=ALL

####### uncomment if modules not already loaded
##### module load gcc
##### module load mpich

set -e
mpicc -O2 -o reduce reduce.c

echo "impl,n,p,local_n,t_local,t_step2,t_total"

REPS=5

for IMPL in naive cube mpi stacked; do
  for LOG_P in 0 1 2 3 4 5; do        # p = 1,2,4,8,16,32
    P=$((1 << LOG_P))
    for LOG_N in $(seq 1 20); do       # n = 2..2^20
      N=$((1 << LOG_N))
      # constraints: n > p  AND  n % p == 0
      if [ $N -le $P ]; then continue; fi
      if [ $((N % P)) -ne 0 ]; then continue; fi
      srun -n $P ./reduce $IMPL $N $REPS
    done
  done
done

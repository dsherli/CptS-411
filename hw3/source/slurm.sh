#!/bin/bash
#SBATCH --job-name=gol
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=32
#SBATCH --export=ALL

module load gcc
module load mpich

set -e

mpicc -O2 -o gol gol.c -lm

G=100

echo "n,p,total_time,avg_gen_time,comm_time,comp_time,comm_pct,comp_pct"

for LOG_N in 2 3 4 5 6 7 8 9 10; do
    N=$((1 << LOG_N))
    for LOG_P in 0 1 2 3 4 5; do
        P=$((1 << LOG_P))
        # n must be divisible by p and n >= p
        if [ $N -ge $P ] && [ $((N % P)) -eq 0 ]; then
            srun -n $P ./gol $N $G
        fi
    done
done

echo "All done."

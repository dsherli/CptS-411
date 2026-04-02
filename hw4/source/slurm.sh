#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

module load gcc

gcc -O2 -fopenmp -o pi pi.c -lm

# clean out old results
rm -f results.csv

# speedup tests - vary p for large n
for n in 1024 1048576 16777216 268435456 1073741824; do
    for p in 1 2 4 8 16; do
        echo "Running n=$n p=$p"
        ./pi $n $p
    done
done

# precision tests - increase n with p=16
for n in 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456 1073741824; do
    echo "Precision test n=$n p=16"
    ./pi $n 16
done

echo "Done, results in results.csv"

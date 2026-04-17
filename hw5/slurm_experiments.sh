#!/bin/bash
#SBATCH --job-name=pagerank_sweep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL

module load gcc

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SCRIPT_DIR"

if [ ! -x ./pagerank_omp ]; then
    echo "Executable not found: $SCRIPT_DIR/pagerank_omp"
    echo "Compile it first from the login node after loading gcc:"
    echo "  module load gcc"
    echo "  gcc -O2 -fopenmp -o pagerank_omp pagerank_omp.c -lm"
    exit 1
fi

GRAPHS="${GRAPHS:-$SCRIPT_DIR/facebook_combined.txt}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8 16}"
K_LIST="${K_LIST:-10 50 100 500 1000}"
D="${D:-0.15}"
REPS="${REPS:-3}"
CSV_FILE="${CSV_FILE:-pagerank_experiments_${SLURM_JOB_ID}.csv}"
TOP5_FILE="${TOP5_FILE:-pagerank_top5_${SLURM_JOB_ID}.txt}"

resolve_mode() {
    local graph_path="$1"
    local graph_name

    graph_name="$(basename "$graph_path")"
    case "$graph_name" in
        facebook_combined.txt)
            echo "undirected"
            ;;
        *)
            echo "directed"
            ;;
    esac
}

extract_value() {
    local file_path="$1"
    local label="$2"
    awk -F': ' -v label="$label" '$1 == label { print $2; exit }' "$file_path"
}

extract_top5() {
    local file_path="$1"
    awk '
        /^Top 5 nodes by estimated PageRank:/ { capture = 1; next }
        capture && /^[1-5]\./ { print; count++; if (count == 5) exit }
    ' "$file_path" | paste -sd ' | ' -
}

printf "graph,mode,k,d,threads,rep,nodes,edges,total_visits,load_time_s,walk_time_s,top1,top2,top3,top4,top5\n" > "$CSV_FILE"
printf "# Detailed top-5 outputs for job %s\n" "${SLURM_JOB_ID:-manual}" > "$TOP5_FILE"

echo "Host: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Executable: $SCRIPT_DIR/pagerank_omp"
echo "Graphs: $GRAPHS"
echo "K values: $K_LIST"
echo "D: $D"
echo "Threads: $THREAD_LIST"
echo "Repetitions: $REPS"
echo "CSV output: $CSV_FILE"
echo "Top-5 log: $TOP5_FILE"
echo

for graph in $GRAPHS; do
    if [ ! -f "$graph" ]; then
        echo "Input graph not found: $graph"
        exit 1
    fi

    mode="$(resolve_mode "$graph")"

    for k in $K_LIST; do
        for threads in $THREAD_LIST; do
            export OMP_NUM_THREADS="$threads"

            rep=1
            while [ "$rep" -le "$REPS" ]; do
                tmp_output="$(mktemp)"

                echo "Running graph=$(basename "$graph") mode=$mode K=$k D=$D threads=$threads rep=$rep"
                srun ./pagerank_omp "$graph" "$k" "$D" "$threads" "$mode" > "$tmp_output"

                nodes="$(extract_value "$tmp_output" "Nodes")"
                edges="$(extract_value "$tmp_output" "Edges stored")"
                total_visits="$(extract_value "$tmp_output" "Total visits")"
                load_time="$(extract_value "$tmp_output" "Graph load/build time" | awk '{print $1}')"
                walk_time="$(extract_value "$tmp_output" "Random walk time" | awk '{print $1}')"

                top1="$(awk '/^1\./ { print; exit }' "$tmp_output")"
                top2="$(awk '/^2\./ { print; exit }' "$tmp_output")"
                top3="$(awk '/^3\./ { print; exit }' "$tmp_output")"
                top4="$(awk '/^4\./ { print; exit }' "$tmp_output")"
                top5="$(awk '/^5\./ { print; exit }' "$tmp_output")"

                printf "\"%s\",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" \
                    "$(basename "$graph")" \
                    "$mode" \
                    "$k" \
                    "$D" \
                    "$threads" \
                    "$rep" \
                    "$nodes" \
                    "$edges" \
                    "$total_visits" \
                    "$load_time" \
                    "$walk_time" \
                    "$top1" \
                    "$top2" \
                    "$top3" \
                    "$top4" \
                    "$top5" >> "$CSV_FILE"

                {
                    echo "graph=$(basename "$graph") mode=$mode K=$k D=$D threads=$threads rep=$rep"
                    cat "$tmp_output"
                    echo
                } >> "$TOP5_FILE"

                rm -f "$tmp_output"
                rep=$((rep + 1))
            done
        done
    done
done

echo
echo "Experiment sweep complete."
echo "CSV written to: $CSV_FILE"
echo "Detailed results written to: $TOP5_FILE"

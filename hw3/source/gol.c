/*
 * gol.c - parallel game of life with MPI
 * does row-wise decomposition, each rank gets n/p rows
 * torus wrapping for the board edges
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIGPRIME 93563
#define ALIVE 1
#define DEAD  0

// globals - set in main
static int rank, nprocs;
static int n;          // board is n x n
static int G;          // how many generations to run
static int local_rows; // n/nprocs

// index into a flat 2d array
static inline int idx(int r, int c, int cols)
{
    return r * cols + c;
}

// wraps column for torus topology
static inline int wrap_col(int c)
{
    return (c + n) % n;
}

/*
 * GenerateInitialGoL - sets up the board
 * rank 0 makes p seeds and scatters them out, then everyone
 * fills their local chunk with random alive/dead cells
 */
static void GenerateInitialGoL(int *grid)
{
    int seed;

    if (rank == 0) {
        // generate p seeds and scatter
        srand(BIGPRIME);
        int *seeds = malloc(nprocs * sizeof(int));
        for (int i = 0; i < nprocs; i++)
            seeds[i] = (rand() % BIGPRIME) + 1;
        MPI_Scatter(seeds, 1, MPI_INT, &seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        free(seeds);
    } else {
        MPI_Scatter(NULL, 1, MPI_INT, &seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // each rank fills its own cells
    srand(seed);
    for (int i = 0; i < local_rows * n; i++) {
        int val = (rand() % BIGPRIME) + 1;
        grid[i] = (val % 2 == 0) ? ALIVE : DEAD;
    }
}

/*
 * count neighbors that are alive around cell (r,c) in the extended grid
 * r,c are in range [1..local_rows] x [0..n-1] since row 0 and
 * local_rows+1 are the ghost rows
 */
static int count_alive_neighbors(const int *ext, int r, int c)
{
    int cnt = 0;
    int cols = n;

    // cardinal directions
    cnt += ext[idx(r-1, wrap_col(c),   cols)];  // N
    cnt += ext[idx(r+1, wrap_col(c),   cols)];  // S
    cnt += ext[idx(r,   wrap_col(c+1), cols)];  // E
    cnt += ext[idx(r,   wrap_col(c-1), cols)];  // W

    // diagonals
    cnt += ext[idx(r-1, wrap_col(c+1), cols)];  // NE
    cnt += ext[idx(r-1, wrap_col(c-1), cols)];  // NW
    cnt += ext[idx(r+1, wrap_col(c+1), cols)];  // SE
    cnt += ext[idx(r+1, wrap_col(c-1), cols)];  // SW

    return cnt;
}

/*
 * DetermineState - figures out if a cell lives or dies
 * 3-5 alive neighbors = alive, otherwise dead
 * (doesnt matter if cell was alive or dead before, same rule)
 */
static int DetermineState(int current, int alive_nbrs)
{
    if (alive_nbrs >= 3 && alive_nbrs <= 5)
        return ALIVE;
    return DEAD;
}

/*
 * exchange_ghosts
 * sends top/bottom real rows to neighboring ranks so everyone
 * has the data they need for the border cells.
 * returns how long the communication took
 *
 * the corner wrapping thing from the assignment spec actually
 * just works out to normal torus behavior - the ghost rows
 * plus wrap_col handles it. I checked all 4 corners and
 * the "diagonally opposite corner" rule gives the same result
 * as standard 2d torus wrapping so no special cases needed
 */
static double exchange_ghosts(int *ext)
{
    double t0 = MPI_Wtime();

    int north = (rank - 1 + nprocs) % nprocs;
    int south = (rank + 1) % nprocs;
    int cols = n;

    // send first row up, recv from below into bottom ghost
    MPI_Sendrecv(&ext[idx(1, 0, cols)],              cols, MPI_INT, north, 0,
                 &ext[idx(local_rows + 1, 0, cols)],  cols, MPI_INT, south, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // send last row down, recv from above into top ghost
    MPI_Sendrecv(&ext[idx(local_rows, 0, cols)], cols, MPI_INT, south, 1,
                 &ext[idx(0, 0, cols)],          cols, MPI_INT, north, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double t1 = MPI_Wtime();
    return t1 - t0;
}

/*
 * Simulate - main loop, runs G generations
 * uses two buffers so we read old state while writing new state
 */
static void Simulate(int *grid, double *comp_time, double *comm_time)
{
    int cols = n;
    int ext_size = (local_rows + 2) * cols;

    int *ext  = malloc(ext_size * sizeof(int));
    int *next = malloc(local_rows * cols * sizeof(int));

    *comp_time = 0.0;
    *comm_time = 0.0;

    for (int gen = 0; gen < G; gen++) {

        MPI_Barrier(MPI_COMM_WORLD);

        // put current grid into the interior of extended grid
        memcpy(&ext[idx(1, 0, cols)], grid, local_rows * cols * sizeof(int));

        // do ghost row exchange
        double ct = exchange_ghosts(ext);
        *comm_time += ct;

        // update all cells
        double tc0 = MPI_Wtime();
        for (int r = 1; r <= local_rows; r++) {
            for (int c = 0; c < cols; c++) {
                int nbrs = count_alive_neighbors(ext, r, c);
                int cur = ext[idx(r, c, cols)];
                next[(r-1) * cols + c] = DetermineState(cur, nbrs);
            }
        }
        double tc1 = MPI_Wtime();
        *comp_time += (tc1 - tc0);

        memcpy(grid, next, local_rows * cols * sizeof(int));
    }

    free(ext);
    free(next);
}

/*
 * DisplayGoL - gather everything to rank 0 and print
 * only really usefull for small boards when debugging
 */
static void DisplayGoL(const int *grid)
{
    int *full = NULL;
    if (rank == 0)
        full = malloc(n * n * sizeof(int));

    MPI_Gather(grid, local_rows * n, MPI_INT,
               full, local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++)
                printf("%c ", full[r * n + c] ? 'O' : '.');
            printf("\n");
        }
        printf("----\n");
        free(full);
    }
}

// quick helper to count total alive cells across all ranks
static int count_total_alive(const int *grid)
{
    int local_alive = 0;
    for (int i = 0; i < local_rows * n; i++)
        local_alive += grid[i];

    int total;
    MPI_Allreduce(&local_alive, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return total;
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <n> <G>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);
    G = atoi(argv[2]);

    if (n % nprocs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: n (%d) must be divisible by p (%d)\n", n, nprocs);
        MPI_Finalize();
        return 1;
    }
    local_rows = n / nprocs;

    int *grid = malloc(local_rows * n * sizeof(int));

    GenerateInitialGoL(grid);

    // show inital board if its small enough to read
    if (n <= 16)
        DisplayGoL(grid);

    int alive_before = count_total_alive(grid);

    // run the simulation
    double comp_time, comm_time;
    double t_start = MPI_Wtime();
    Simulate(grid, &comp_time, &comm_time);
    double t_end = MPI_Wtime();

    double total_time = t_end - t_start;

    if (n <= 16)
        DisplayGoL(grid);

    int alive_after = count_total_alive(grid);

    // get max times across ranks for reporting
    double max_comm;
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double max_total;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double my_comp = total_time - comm_time;
    double max_comp;
    MPI_Reduce(&my_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("n=%d, G=%d, p=%d\n", n, G, nprocs);
        printf("Alive: before=%d, after=%d\n", alive_before, alive_after);
        printf("Total runtime:        %.6f s\n", max_total);
        printf("Avg time/generation:  %.6f s\n", max_total / G);
        printf("Total comm time:      %.6f s\n", max_comm);
        printf("Total comp time:      %.6f s\n", max_total - max_comm);
        printf("Comm %%:               %.2f%%\n", 100.0 * max_comm / max_total);
        printf("Comp %%:               %.2f%%\n", 100.0 * (max_total - max_comm) / max_total);

        // write to CSV for analysis
        FILE *csv = fopen("results.csv", "a");
        if (csv) {
            // check if file is empty (first write) to add headers
            fseek(csv, 0, SEEK_END);
            long size = ftell(csv);
            if (size == 0) {
                fprintf(csv, "n,G,p,total_runtime,avg_time_per_gen,comm_time,comp_time,comm_pct,comp_pct\n");
            }
            // append data row
            fprintf(csv, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f\n",
                    n, G, nprocs, max_total, max_total / G, max_comm,
                    max_total - max_comm,
                    100.0 * max_comm / max_total,
                    100.0 * (max_total - max_comm) / max_total);
            fclose(csv);
        }
    }

    free(grid);
    MPI_Finalize();
    return 0;
}

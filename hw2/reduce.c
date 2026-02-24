#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

// implement these
static int NaiveAllReduce_IntSum(int local_sum, MPI_Comm comm) {
  (void)local_sum;
  (void)comm;
  return 0;
}

static int HypercubicAllReduce_IntSum(int local_sum, MPI_Comm comm) {
  (void)local_sum;
  (void)comm;
  return 0;
}

static void StackedReduce_IntSum(int *local_vec, int local_n, MPI_Comm comm) {
  (void)local_vec;
  (void)local_n;
  (void)comm;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  // args: impl n reps
  const char *impl = (argc > 1) ? argv[1] : "mpi";
  int n = (argc > 2) ? atoi(argv[2]) : (1 << 20);
  int reps = (argc > 3) ? atoi(argv[3]) : 5;

  if (!is_pow2(p)) {
    if (rank == 0)
      fprintf(stderr, "p must be power of 2\n");
    MPI_Abort(comm, 1);
  }
  if (n <= p) {
    if (rank == 0)
      fprintf(stderr, "need n > p\n");
    MPI_Abort(comm, 1);
  }
  if (n % p != 0) {
    if (rank == 0)
      fprintf(stderr, "need n multiple of p\n");
    MPI_Abort(comm, 1);
  }

  int local_n = n / p;
  int *local = (int *)malloc((size_t)local_n * sizeof(int));
  if (!local) {
    if (rank == 0)
      fprintf(stderr, "malloc failed\n");
    MPI_Abort(comm, 1);
  }

  // best of reps timing (max across ranks)
  double best_local = 1e300, best_step2 = 1e300, best_total = 1e300;

  for (int r = 0; r < reps; r++) {
    // generate data (not timed)
    unsigned int seed = (unsigned int)(12345 + 10007 * rank + 97 * r);
    for (int i = 0; i < local_n; i++) {
      seed = 1664525u * seed + 1013904223u;
      local[i] = (int)(seed % 100u) + 1;
    }

    // step 1: local sum
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    int local_sum = 0;
    for (int i = 0; i < local_n; i++)
      local_sum += local[i];

    double t1 = MPI_Wtime();
    MPI_Barrier(comm);

    double t_local = t1 - t0;

    // step 2: all-reduce / stacked reduce
    double t2 = MPI_Wtime();

    if (strcmp(impl, "stacked") == 0) {
      StackedReduce_IntSum(local, local_n, comm);
      // correctness oracle for vector
      int *oracle = (int *)malloc((size_t)local_n * sizeof(int));
      if (!oracle) {
        if (rank == 0)
          fprintf(stderr, "malloc failed\n");
        MPI_Abort(comm, 1);
      }
      MPI_Allreduce(local, oracle, local_n, MPI_INT, MPI_SUM, comm);
      for (int i = 0; i < local_n; i++) {
        if (local[i] != oracle[i]) {
          fprintf(stderr, "Rank %d FAIL stacked at i=%d got=%d want=%d\n", rank,
                  i, local[i], oracle[i]);
          MPI_Abort(comm, 2);
        }
      }
      free(oracle);
    } else {
      int computed = 0;
      if (strcmp(impl, "naive") == 0)
        computed = NaiveAllReduce_IntSum(local_sum, comm);
      else if (strcmp(impl, "cube") == 0)
        computed = HypercubicAllReduce_IntSum(local_sum, comm);
      else
        computed = 0, MPI_Allreduce(&local_sum, &computed, 1, MPI_INT, MPI_SUM,
                                    comm); // "mpi" default

      // correctness oracle
      int oracle = 0;
      MPI_Allreduce(&local_sum, &oracle, 1, MPI_INT, MPI_SUM, comm);
      if (computed != oracle) {
        fprintf(stderr, "Rank %d FAIL scalar got=%d want=%d\n", rank, computed,
                oracle);
        MPI_Abort(comm, 2);
      }
    }

    MPI_Barrier(comm);
    double t3 = MPI_Wtime();

    double t_step2 = t3 - t2;
    double t_total = t_local + t_step2;

    double max_local, max_step2, max_total;
    MPI_Allreduce(&t_local, &max_local, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&t_step2, &max_step2, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&t_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (max_local < best_local)
      best_local = max_local;
    if (max_step2 < best_step2)
      best_step2 = max_step2;
    if (max_total < best_total)
      best_total = max_total;
  }

  if (rank == 0) {
    printf("%s,%d,%d,%d,%.9f,%.9f,%.9f\n", impl, n, p, local_n, best_local,
           best_step2, best_total);
    fflush(stdout);
  }

  free(local);
  MPI_Finalize();
  return 0;
}

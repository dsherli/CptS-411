#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

// naive all-reduce using a two-pass linear pipeline
// pass 1 sweeps left to right so rank p-1 ends up with the total
// pass 2 sweeps right to left so everyone gets the total
//
// we use Send/Recv instead of Sendrecv here on purpose - with Sendrecv
// each rank would blast out its local value before receiving anything,
// so the running sum from the left never makes it in. kills the pipeline.
static int NaiveAllReduce_IntSum(int local_sum, MPI_Comm comm)
{
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  if (p == 1)
    return local_sum;

  int val = local_sum;

  // pass 1: chain from rank 0 up to rank p-1, accumulating as we go
  if (rank == 0)
  {
    MPI_Send(&val, 1, MPI_INT, 1, 0, comm);
  }
  else
  {
    int recv;
    MPI_Recv(&recv, 1, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
    val += recv;
    if (rank < p - 1)
      MPI_Send(&val, 1, MPI_INT, rank + 1, 0, comm);
  }
  // at this point rank p-1 has the global sum

  int global_sum = val;

  // pass 2: chain back from rank p-1 down to rank 0
  if (rank == p - 1)
  {
    MPI_Send(&global_sum, 1, MPI_INT, rank - 1, 1, comm);
  }
  else
  {
    MPI_Recv(&global_sum, 1, MPI_INT, rank + 1, 1, comm, MPI_STATUS_IGNORE);
    if (rank > 0)
      MPI_Send(&global_sum, 1, MPI_INT, rank - 1, 1, comm);
  }

  return global_sum;
}

// hypercubic all-reduce - does the job in lg(p) rounds instead of O(p)
// each round d, flip bit d of your rank to find your partner and swap values
// after lg(p) swaps everyone has seen everything and holds the global sum
static int HypercubicAllReduce_IntSum(int local_sum, MPI_Comm comm)
{
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  int val = local_sum;
  for (int d = 0; (1 << d) < p; d++)
  {
    int partner = rank ^ (1 << d);
    int recv;
    MPI_Sendrecv(&val, 1, MPI_INT, partner, d, &recv, 1, MPI_INT, partner, d,
                 comm, MPI_STATUS_IGNORE);
    val += recv;
  }
  return val;
}

// same hypercubic idea but now each process holds an array instead of one value
// every round we swap the full array with our partner and add element-wise
// after lg(p) rounds local_vec[i] is the sum of local_vec[i] across all ranks
static void StackedReduce_IntSum(int *local_vec, int local_n, MPI_Comm comm)
{
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  int *recv_buf = (int *)malloc((size_t)local_n * sizeof(int));
  if (!recv_buf)
    MPI_Abort(comm, 1);

  for (int d = 0; (1 << d) < p; d++)
  {
    int partner = rank ^ (1 << d);
    MPI_Sendrecv(local_vec, local_n, MPI_INT, partner, d, recv_buf, local_n,
                 MPI_INT, partner, d, comm, MPI_STATUS_IGNORE);
    for (int i = 0; i < local_n; i++)
      local_vec[i] += recv_buf[i];
  }

  free(recv_buf);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  // usage: ./reduce <impl> <n> <reps>
  const char *impl = (argc > 1) ? argv[1] : "mpi";
  int n = (argc > 2) ? atoi(argv[2]) : (1 << 20);
  int reps = (argc > 3) ? atoi(argv[3]) : 5;

  if (!is_pow2(p))
  {
    if (rank == 0)
      fprintf(stderr, "p must be power of 2\n");
    MPI_Abort(comm, 1);
  }
  if (n <= p)
  {
    if (rank == 0)
      fprintf(stderr, "need n > p\n");
    MPI_Abort(comm, 1);
  }
  if (n % p != 0)
  {
    if (rank == 0)
      fprintf(stderr, "need n multiple of p\n");
    MPI_Abort(comm, 1);
  }

  int local_n = n / p;
  int *local = (int *)malloc((size_t)local_n * sizeof(int));
  if (!local)
  {
    if (rank == 0)
      fprintf(stderr, "malloc failed\n");
    MPI_Abort(comm, 1);
  }

  // track best (minimum) time across all reps
  double best_local = 1e300, best_step2 = 1e300, best_total = 1e300;

  for (int r = 0; r < reps; r++)
  {
    // fill local array with random ints 1-100, not counted in timing
    unsigned int seed = (unsigned int)(12345 + 10007 * rank + 97 * r);
    for (int i = 0; i < local_n; i++)
    {
      seed = 1664525u * seed + 1013904223u;
      local[i] = (int)(seed % 100u) + 1;
    }

    // time step 1: compute local sum
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    int local_sum = 0;
    for (int i = 0; i < local_n; i++)
      local_sum += local[i];

    double t1 = MPI_Wtime();
    MPI_Barrier(comm);

    double t_local = t1 - t0;

    // time step 2: run whichever all-reduce we're testing
    double t2 = MPI_Wtime();

    if (strcmp(impl, "stacked") == 0)
    {
      // need to snapshot local before the reduce modifies it in place,
      // otherwise the oracle MPI_Allreduce would run on already-summed data
      // and give p*correct_answer instead of correct_answer
      int *orig = (int *)malloc((size_t)local_n * sizeof(int));
      if (!orig)
      {
        if (rank == 0)
          fprintf(stderr, "malloc failed\n");
        MPI_Abort(comm, 1);
      }
      memcpy(orig, local, (size_t)local_n * sizeof(int));

      StackedReduce_IntSum(local, local_n, comm);

      // check against MPI's built-in answer
      int *oracle = (int *)malloc((size_t)local_n * sizeof(int));
      if (!oracle)
      {
        if (rank == 0)
          fprintf(stderr, "malloc failed\n");
        MPI_Abort(comm, 1);
      }
      MPI_Allreduce(orig, oracle, local_n, MPI_INT, MPI_SUM, comm);
      free(orig);
      for (int i = 0; i < local_n; i++)
      {
        if (local[i] != oracle[i])
        {
          fprintf(stderr, "Rank %d FAIL stacked at i=%d got=%d want=%d\n", rank,
                  i, local[i], oracle[i]);
          MPI_Abort(comm, 2);
        }
      }
      free(oracle);
    }
    else
    {
      int computed = 0;
      if (strcmp(impl, "naive") == 0)
        computed = NaiveAllReduce_IntSum(local_sum, comm);
      else if (strcmp(impl, "cube") == 0)
        computed = HypercubicAllReduce_IntSum(local_sum, comm);
      else
        computed = 0, MPI_Allreduce(&local_sum, &computed, 1, MPI_INT, MPI_SUM,
                                    comm); // "mpi" default

      // verify against MPI's answer
      int oracle = 0;
      MPI_Allreduce(&local_sum, &oracle, 1, MPI_INT, MPI_SUM, comm);
      if (computed != oracle)
      {
        fprintf(stderr, "Rank %d FAIL scalar got=%d want=%d\n", rank, computed,
                oracle);
        MPI_Abort(comm, 2);
      }
    }

    MPI_Barrier(comm);
    double t3 = MPI_Wtime();

    double t_step2 = t3 - t2;
    double t_total = t_local + t_step2;

    // use the slowest rank's time so we capture the real wall time
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

  if (rank == 0)
  {
    printf("%s,%d,%d,%d,%.9f,%.9f,%.9f\n", impl, n, p, local_n, best_local,
           best_step2, best_total);
    fflush(stdout);
  }

  free(local);
  MPI_Finalize();
  return 0;
}

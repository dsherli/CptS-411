/*
 * pi.c - estimate pi by throwing darts at a unit square
 * uses openmp for multithreading
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <n> <p>\n", argv[0]);
        return 1;
    }

    long long n = atoll(argv[1]); // number of darts
    int p = atoi(argv[2]);        // number of threads

    omp_set_num_threads(p);

    long long total_in_circle = 0;

    double t_start = omp_get_wtime();

    // each thread throws n/p darts (roughly), uses its own seed
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // seed each thread differently
        unsigned int seed = 42 + tid * 1013;

        long long local_count = 0;

        #pragma omp for schedule(static)
        for (long long i = 0; i < n; i++) {
            // rand_r is thread safe unlike rand
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;

            if (x*x + y*y <= 1.0)
                local_count++;
        }

        // add to shared counter
        #pragma omp atomic
        total_in_circle += local_count;
    }

    double t_end = omp_get_wtime();

    // pi/4 = fraction of darts inside circle
    double pi_est = 4.0 * (double)total_in_circle / (double)n;
    double elapsed = t_end - t_start;

    printf("n=%lld, p=%d\n", n, p);
    printf("Pi estimate: %.20f\n", pi_est);
    printf("Error:       %.20f\n", fabs(pi_est - M_PI));
    printf("Time:        %.6f s\n", elapsed);

    // append to csv for graphing later
    FILE *csv = fopen("results.csv", "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        long size = ftell(csv);
        if (size == 0)
            fprintf(csv, "n,p,pi_estimate,error,time\n");
        fprintf(csv, "%lld,%d,%.20f,%.20f,%.6f\n",
                n, p, pi_est, fabs(pi_est - M_PI), elapsed);
        fclose(csv);
    }

    return 0;
}

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Hybrid MPI + OpenMP Monte Carlo PageRank estimator.
 *
 * Compile for distributed memory with:
 *   mpicc -O2 -fopenmp pagerank_omp_final.c -o pagerank_mpi_omp
 *
 * Each rank builds a local CSR copy of the graph, owns a strided subset of
 * starting vertices, accumulates rank-local visit counters, and reduces those
 * counters to rank 0 for final top-5 output.
 */

#ifndef HAVE_MPI
#if defined(__has_include)
#if __has_include(<mpi.h>)
#define HAVE_MPI 1
#else
#define HAVE_MPI 0
#endif
#else
#define HAVE_MPI 0
#endif
#endif

#if HAVE_MPI
#include <mpi.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#else
static double omp_get_wtime(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void omp_set_num_threads(int threads)
{
    (void)threads;
}

static int omp_get_thread_num(void)
{
    return 0;
}

static int omp_get_num_threads(void)
{
    return 1;
}
#endif

#if HAVE_MPI
static int g_mpi_initialized = 0;
#endif

typedef struct
{
    uint32_t src;
    uint32_t dst;
} Edge;

typedef struct
{
    uint64_t state;
} RngState;

static void die(const char *message)
{
    fprintf(stderr, "Error: %s\n", message);
#if HAVE_MPI
    if (g_mpi_initialized)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr)
        die("out of memory");
    return ptr;
}

static void *xcalloc(size_t count, size_t size)
{
    void *ptr = calloc(count, size);
    if (!ptr)
        die("out of memory");
    return ptr;
}

static int has_suffix(const char *text, const char *suffix)
{
    size_t text_len = strlen(text);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > text_len)
        return 0;
    return strcmp(text + text_len - suffix_len, suffix) == 0;
}

static uint64_t splitmix64_next(uint64_t *state)
{
    uint64_t z;

    *state += 0x9e3779b97f4a7c15ULL;
    z = *state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_seed(RngState *rng, uint64_t seed)
{
    rng->state = seed;
    if (rng->state == 0)
        rng->state = 0x6a09e667f3bcc909ULL;
}

static uint64_t rng_u64(RngState *rng)
{
    return splitmix64_next(&rng->state);
}

static double rng_unit(RngState *rng)
{
    return (rng_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static uint32_t rng_bounded(RngState *rng, uint32_t bound)
{
    uint64_t threshold;
    uint64_t value;

    if (bound == 0)
        return 0;

    threshold = (uint64_t)(-bound) % bound;
    for (;;)
    {
        value = rng_u64(rng);
        if (value >= threshold)
            return (uint32_t)(value % bound);
    }
}

static uint64_t parse_u64(const char *text, const char *label)
{
    char *endptr = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &endptr, 10);
    if (errno != 0 || endptr == text || *endptr != '\0')
    {
        fprintf(stderr, "Error: invalid %s '%s'\n", label, text);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static double parse_double(const char *text, const char *label)
{
    char *endptr = NULL;
    double value;

    errno = 0;
    value = strtod(text, &endptr);
    if (errno != 0 || endptr == text || *endptr != '\0')
    {
        fprintf(stderr, "Error: invalid %s '%s'\n", label, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static int parse_graph_mode(const char *text)
{
    if (strcmp(text, "directed") == 0)
        return 0;
    if (strcmp(text, "undirected") == 0)
        return 1;

    fprintf(stderr, "Error: graph mode must be 'directed' or 'undirected', got '%s'\n", text);
    exit(EXIT_FAILURE);
}

static double wall_time(void)
{
#if HAVE_MPI
    if (g_mpi_initialized)
        return MPI_Wtime();
#endif
    return omp_get_wtime();
}

static void maybe_grow_edges(Edge **edges, size_t *capacity, size_t needed)
{
    Edge *resized;
    size_t new_capacity;

    if (needed <= *capacity)
        return;

    new_capacity = (*capacity == 0) ? 1024 : *capacity;
    while (new_capacity < needed)
        new_capacity *= 2;

    resized = realloc(*edges, new_capacity * sizeof(*resized));
    if (!resized)
        die("out of memory");

    *edges = resized;
    *capacity = new_capacity;
}

static void load_edges(const char *path,
                       int undirected,
                       Edge **edges_out,
                       size_t *edge_count_out,
                       uint32_t *node_count_out)
{
    FILE *fp;
    Edge *edges = NULL;
    size_t edge_count = 0;
    size_t capacity = 0;
    char line[256];
    uint32_t max_node_id = 0;
    uint32_t header_nodes = 0;
    int saw_edge = 0;

    fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: failed to open '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    while (fgets(line, sizeof(line), fp))
    {
        char *cursor = line;
        unsigned long src_ul;
        unsigned long dst_ul;
        int scanned;

        while (isspace((unsigned char)*cursor))
            cursor++;

        if (*cursor == '\0')
            continue;

        if (*cursor == '#')
        {
            unsigned long header_ul;
            if (sscanf(cursor, "# Nodes: %lu", &header_ul) == 1)
                header_nodes = (uint32_t)header_ul;
            continue;
        }

        scanned = sscanf(cursor, "%lu %lu", &src_ul, &dst_ul);
        if (scanned != 2)
            continue;

        if (src_ul > UINT32_MAX || dst_ul > UINT32_MAX)
            die("node id exceeds uint32 range");

        maybe_grow_edges(&edges, &capacity, edge_count + (size_t)(undirected ? 2 : 1));
        edges[edge_count].src = (uint32_t)src_ul;
        edges[edge_count].dst = (uint32_t)dst_ul;
        edge_count++;
        saw_edge = 1;

        if (undirected && src_ul != dst_ul)
        {
            edges[edge_count].src = (uint32_t)dst_ul;
            edges[edge_count].dst = (uint32_t)src_ul;
            edge_count++;
        }

        if (src_ul > max_node_id)
            max_node_id = (uint32_t)src_ul;
        if (dst_ul > max_node_id)
            max_node_id = (uint32_t)dst_ul;
    }

    fclose(fp);

    if (header_nodes > 0)
        *node_count_out = (header_nodes > max_node_id + 1U) ? header_nodes : (max_node_id + 1U);
    else if (saw_edge)
        *node_count_out = max_node_id + 1U;
    else
        *node_count_out = 0;

    *edges_out = edges;
    *edge_count_out = edge_count;
}

static void build_csr(const Edge *edges,
                      size_t edge_count,
                      uint32_t node_count,
                      uint64_t **offsets_out,
                      uint32_t **neighbors_out)
{
    uint64_t *offsets;
    uint32_t *neighbors;
    uint64_t *cursor;
    size_t i;

    offsets = xcalloc((size_t)node_count + 1, sizeof(*offsets));
    neighbors = xmalloc(edge_count * sizeof(*neighbors));

    for (i = 0; i < edge_count; i++)
        offsets[edges[i].src + 1]++;

    for (i = 1; i <= node_count; i++)
        offsets[i] += offsets[i - 1];

    cursor = xmalloc(((size_t)node_count + 1) * sizeof(*cursor));
    memcpy(cursor, offsets, ((size_t)node_count + 1) * sizeof(*cursor));

    for (i = 0; i < edge_count; i++)
        neighbors[cursor[edges[i].src]++] = edges[i].dst;

    free(cursor);

    *offsets_out = offsets;
    *neighbors_out = neighbors;
}

typedef struct
{
    uint32_t node;
    uint64_t visits;
} RankedNode;

static int ranked_node_cmp(const void *lhs, const void *rhs)
{
    const RankedNode *a = (const RankedNode *)lhs;
    const RankedNode *b = (const RankedNode *)rhs;

    if (a->visits < b->visits)
        return 1;
    if (a->visits > b->visits)
        return -1;
    if (a->node > b->node)
        return 1;
    if (a->node < b->node)
        return -1;
    return 0;
}

#if HAVE_MPI
static void reduce_visit_counts(const uint64_t *local_counts,
                                uint64_t *global_counts,
                                uint32_t node_count)
{
    uint64_t offset = 0;

    while (offset < node_count)
    {
        uint64_t remaining = (uint64_t)node_count - offset;
        int chunk = (remaining > (uint64_t)INT_MAX) ? INT_MAX : (int)remaining;

        MPI_Reduce(local_counts + offset,
                   global_counts ? global_counts + offset : NULL,
                   chunk,
                   MPI_UINT64_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);
        offset += (uint64_t)chunk;
    }
}
#endif

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: mpirun -np <ranks> %s <graph_file> <walk_length_K> <damping_ratio_D> <threads_per_rank> [directed|undirected]\n",
            prog);
    fprintf(stderr,
            "Default graph mode is 'undirected' for files ending in 'facebook_combined.txt', otherwise 'directed'.\n");
#if !HAVE_MPI
    fprintf(stderr,
            "This build did not find mpi.h, so it will run as one MPI rank. Compile with mpicc for distributed execution.\n");
#endif
}

int main(int argc, char **argv)
{
    const char *graph_path;
    uint64_t walk_length;
    double damping_ratio;
    int thread_count;
    int undirected = 0;
    Edge *edges = NULL;
    size_t edge_count = 0;
    uint32_t node_count = 0;
    uint64_t *offsets = NULL;
    uint32_t *neighbors = NULL;
    uint64_t *visit_counts = NULL;
    uint64_t *global_visit_counts = NULL;
    uint64_t total_visits = 0;
    RankedNode top[5];
    double t0;
    double t1;
    double t2;
    double local_load_time;
    double local_walk_time;
    double load_time;
    double walk_time;
    size_t i;
    int rank = 0;
    int ranks = 1;

#if HAVE_MPI
    MPI_Init(&argc, &argv);
    g_mpi_initialized = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#endif

    if (argc < 5 || argc > 6)
    {
#if HAVE_MPI
        if (rank == 0)
            usage(argv[0]);
        MPI_Finalize();
        g_mpi_initialized = 0;
#else
        usage(argv[0]);
#endif
        return EXIT_FAILURE;
    }

    graph_path = argv[1];
    walk_length = parse_u64(argv[2], "walk length");
    damping_ratio = parse_double(argv[3], "damping ratio");
    thread_count = (int)parse_u64(argv[4], "thread count");

    if (damping_ratio < 0.0 || damping_ratio > 1.0)
        die("damping ratio must be between 0 and 1");
    if (thread_count <= 0)
        die("thread count must be positive");

    if (argc == 6)
        undirected = parse_graph_mode(argv[5]);
    else if (has_suffix(graph_path, "facebook_combined.txt"))
        undirected = 1;

    t0 = wall_time();
    load_edges(graph_path, undirected, &edges, &edge_count, &node_count);
    if (node_count == 0)
        die("graph has no nodes");
    build_csr(edges, edge_count, node_count, &offsets, &neighbors);
    free(edges);
    t1 = wall_time();

    visit_counts = xcalloc(node_count, sizeof(*visit_counts));
    omp_set_num_threads(thread_count);

#if HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = wall_time();
#endif

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        uint64_t *local_counts = xcalloc(node_count, sizeof(*local_counts));
        RngState rng;
        uint64_t node;
        uint64_t seed = (uint64_t)time(NULL) ^
                        (0x9e3779b97f4a7c15ULL * (uint64_t)(rank + 1)) ^
                        (0xbf58476d1ce4e5b9ULL * (uint64_t)(tid + 1));

        rng_seed(&rng, seed);

        for (node = (uint64_t)rank + (uint64_t)ranks * (uint64_t)tid;
             node < node_count;
             node += (uint64_t)ranks * (uint64_t)threads)
        {
            uint32_t current = (uint32_t)node;
            uint64_t step;

            local_counts[current]++;

            for (step = 0; step < walk_length; step++)
            {
                uint64_t begin = offsets[current];
                uint64_t end = offsets[current + 1];
                uint64_t degree = end - begin;

                if (degree == 0 || rng_unit(&rng) < damping_ratio)
                {
                    current = rng_bounded(&rng, node_count);
                }
                else
                {
                    uint32_t pick = rng_bounded(&rng, (uint32_t)degree);
                    current = neighbors[begin + pick];
                }

                local_counts[current]++;
            }
        }

#pragma omp critical
        {
            uint32_t v;
            for (v = 0; v < node_count; v++)
                visit_counts[v] += local_counts[v];
        }

        free(local_counts);
    }

    t2 = wall_time();

    local_load_time = t1 - t0;
    local_walk_time = t2 - t1;
    load_time = local_load_time;
    walk_time = local_walk_time;

#if HAVE_MPI
    if (rank == 0)
        global_visit_counts = xcalloc(node_count, sizeof(*global_visit_counts));

    reduce_visit_counts(visit_counts, global_visit_counts, node_count);
    MPI_Reduce(&local_load_time, &load_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_walk_time, &walk_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    global_visit_counts = visit_counts;
#endif

    if (rank != 0)
    {
        free(offsets);
        free(neighbors);
        free(visit_counts);
#if HAVE_MPI
        MPI_Finalize();
        g_mpi_initialized = 0;
#endif
        return EXIT_SUCCESS;
    }

    for (i = 0; i < 5; i++)
    {
        top[i].node = 0;
        top[i].visits = 0;
    }

    for (i = 0; i < node_count; i++)
    {
        total_visits += global_visit_counts[i];
        if (i < 5)
        {
            top[i].node = (uint32_t)i;
            top[i].visits = global_visit_counts[i];
        }
    }

    qsort(top, 5, sizeof(top[0]), ranked_node_cmp);

    for (i = 5; i < node_count; i++)
    {
        RankedNode candidate;

        candidate.node = (uint32_t)i;
        candidate.visits = global_visit_counts[i];
        if (ranked_node_cmp(&candidate, &top[4]) < 0)
        {
            top[4] = candidate;
            qsort(top, 5, sizeof(top[0]), ranked_node_cmp);
        }
    }

    printf("Input graph: %s\n", graph_path);
    printf("Graph mode: %s\n", undirected ? "undirected" : "directed");
    printf("Nodes: %" PRIu32 "\n", node_count);
    printf("Edges stored: %zu\n", edge_count);
    printf("Walk length K: %" PRIu64 "\n", walk_length);
    printf("Damping ratio D: %.6f\n", damping_ratio);
    printf("MPI ranks: %d\n", ranks);
    printf("Threads per rank: %d\n", thread_count);
#if HAVE_MPI
    printf("Execution mode: MPI + OpenMP\n");
#else
    printf("Execution mode: single-rank OpenMP fallback (compile with mpicc for MPI)\n");
#endif
    printf("Graph load/build time: %.6f seconds\n", load_time);
    printf("Random walk time: %.6f seconds\n", walk_time);
    printf("Total visits: %" PRIu64 "\n", total_visits);
    printf("Top 5 nodes by estimated PageRank:\n");

    for (i = 0; i < 5 && i < node_count; i++)
    {
        double pagerank = (total_visits == 0) ? 0.0 : ((double)top[i].visits / (double)total_visits);
        printf("%zu. node=%" PRIu32 " visits=%" PRIu64 " pagerank=%.12f\n",
               i + 1,
               top[i].node,
               top[i].visits,
               pagerank);
    }

    free(offsets);
    free(neighbors);
    if (global_visit_counts != visit_counts)
        free(global_visit_counts);
    free(visit_counts);
#if HAVE_MPI
    MPI_Finalize();
    g_mpi_initialized = 0;
#endif
    return EXIT_SUCCESS;
}

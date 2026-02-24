/*	Cpt S 411, Introduction to Parallel Computing
 *	School of Electrical Engineering and Computer Science
 *
 *	Example code
 *	Send receive test:
 *   	Rank 1 sends to rank 0 (all other ranks, if any, sit idle)
 *	Payload is just one integer in this example, but the code can be adapted to send other types of messages.
 *   	For timing, this code uses C gettimeofday(). Alternatively you can also use MPI_Wtime().
 * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>

#define MIN_MSG_SIZE 1		   // minimum message size in bytes
#define MAX_MSG_SIZE (1 << 20) // maximum message size in bytes (1MB)
#define NUM_ITERATIONS 100	   // number of iterations for averaging
#define OUTPUT_FILE "res.csv"  // output file for results

double get_time_us()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

int main(int argc, char *argv[])
{
	int rank, p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	printf("my rank=%d\n", rank);
	printf("Rank=%d: number of processes =%d\n", rank, p);

	assert(p == 2);

	FILE *outfile = NULL;
	if (rank == 0)
	{
		outfile = fopen(OUTPUT_FILE, "w");
		if (!outfile)
		{
			fprintf(stderr, "Error: could not open output file %s\n", OUTPUT_FILE);
			MPI_Finalize();
			return 1;
		}

		fprintf(outfile, "messages_size_bytes, average_send_time_us, average_recv_time_us,rtt_us,bandwidth_mbps\n");
	}

	char *send_buf = (char *)malloc((MAX_MSG_SIZE)); // allocate buffer for sending messages up to 1MB
	char *recv_buf = (char *)malloc((MAX_MSG_SIZE)); // allocate buffer for receiving messages up to 1MB

	memset(send_buf, 0, MAX_MSG_SIZE); // initialize send buffer to zero
	memset(recv_buf, 0, MAX_MSG_SIZE); // initialize receive buffer to zero

	MPI_Status status;

	double min_rtt = 1e9;		 // initialize min RTT to a large value
	double max_bandwidth = 0.0;	 // initialize max bandwidth to zero
	int buffer_size_est = 0;	 // variable to store estimated buffer size
	double prev_send_time = 0.0; // variable to store previous send time for buffer size estimation
	int buffer_detected = 0;	 // flag to indicate if buffer size has been detected

	// loop over message sizes from 1 byte to 1MB, doubling each time
	for (int m = MIN_MSG_SIZE; m <= MAX_MSG_SIZE; m *= 2)
	{
		double send_time = 0.0;
		double recv_time = 0.0;
		double total_rtt = 0.0;

		MPI_Barrier(MPI_COMM_WORLD); // synchronize before starting timing

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			double t_start, t_post_send, t_post_recv;

			if (rank == 0)
			{
				t_start = get_time_us();
				MPI_Send(send_buf, m, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
				t_post_send = get_time_us();
				MPI_Recv(recv_buf, m, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
				t_post_recv = get_time_us();

				send_time += (t_post_send - t_start);
				recv_time += (t_post_recv - t_post_send);
				total_rtt += (t_post_recv - t_start);
			}
			else
			{
				t_start = get_time_us();
				MPI_Recv(recv_buf, m, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
				t_post_recv = get_time_us();
				MPI_Send(send_buf, m, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
				t_post_send = get_time_us();

				recv_time += (t_post_recv - t_start);
				send_time += (t_post_send - t_post_recv);
			}
		}

		double send_avg = send_time / NUM_ITERATIONS;
		double recv_avg = recv_time / NUM_ITERATIONS;
		double rtt_avg = total_rtt / NUM_ITERATIONS;

		double bandwidth_mbps = 0.0;
		if (rtt_avg > 0)
		{
			bandwidth_mbps = (2.0 * m) / rtt_avg; // bandwidth in bytes per microsecond
		}

		if (rank == 0)
		{
			fprintf(outfile, "%d, %.2f, %.2f, %.2f, %.2f\n", m, send_avg, recv_avg, rtt_avg, bandwidth_mbps);

			if (m >= 64 && rtt_avg < min_rtt)
			{
				min_rtt = rtt_avg;
			}

			if (bandwidth_mbps > max_bandwidth)
			{
				max_bandwidth = bandwidth_mbps;
			}

			if (!buffer_detected && prev_send_time > 0 && send_avg > prev_send_time * 1.5 && m >= 1024)
			{
				buffer_size_est = m / 2; // estimate buffer size as half of the current message size
				buffer_detected = 1;
			}

			prev_send_time = send_avg;
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (rank == 0)
	{
		double latency_est = min_rtt / 2.0;
		double bandwidth_est = max_bandwidth;

		fprintf(outfile, "\nEstimated latency: %.2f microseconds\n", latency_est);
		fprintf(outfile, "Estimated bandwidth: %.2f MB/s\n", bandwidth_est);
		if (buffer_detected)
		{
			fprintf(outfile, "Estimated buffer size: %d bytes\n", buffer_size_est);
		}
		else
		{
			fprintf(outfile, "Buffer size estimation inconclusive\n");
		}

		fclose(outfile);
	}

	free(send_buf);
	free(recv_buf);

	MPI_Finalize();
	return 0;
}

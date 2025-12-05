#include <stdio.h>
#include <stdlib.h>
#include "../include/timer.h"

void calc_sizes(int comm_sz, int rows, int cols, int* obj_sizes);
void calc_displs();

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "ERROR! Not enough arguments\nUsage: mpiexec -n [PROCESS QTY] task2_rows [ROWS] [COLUMNS]\n");
        return 1;
    }

    int rows, columns;
    int comm_sz, my_rank;

    rows = atoi(argv[1]);
    columns = atoi(argv[2]);

    // Initialized MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); // process count
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // current process

    // size per each process
    int* matrix_sizes = calloc(comm_sz, sizeof(int));
    int* vector_sizes = calloc(comm_sz, sizeof(int));

    // displacements for each process
    int* matrix_displs = calloc(comm_sz, sizeof(int));
    int* vector_displs = calloc(comm_sz, sizeof(int)); 



    Timer timer;
    timer_start(&timer);

    MPI_Barrier(MPI_COMM_WORLD);

    timer_stop(&timer);
    double elapsed = time_elapsed(&timer);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Process %d: Elapsed time: %f seconds\n", rank, elapsed);

    MPI_Finalize();
    return 0;
}

// calculate sizes for each process to handle
void calc_sizes(int comm_sz, int rows, int cols, int* obj_sizes) {
    for (int i = 0; i < comm_sz; i++) {
        obj_sizes[i] = (rows / comm_sz + (rows % comm_sz > i ? 1 : 0)) * cols;
    }
}

// calculate displacements indicating starting point for each process
void calc_displs();
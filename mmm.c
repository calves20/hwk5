#include "mmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

// Global variables for matrix dimensions and the number of threads in parallel execution.
unsigned int size, num_threads;
double **A, **B, **SEQ_MATRIX, **PAR_MATRIX;

// Structure for passing multiple arguments to the parallel worker function.
typedef struct {
    int start_row, end_row; // Start and end indices of rows for this thread to process.
    double **A, **B, **C;   // Pointers to the matrices involved in the multiplication.
} WorkerArgs;

// Allocate memory for a square matrix of dimension n x n.
double **matrix_allocate(int n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)malloc(n * sizeof(double));
    }
    return matrix;
}

// Fill a matrix with random values between 0 and 99.
void matrix_fill(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

// Set all elements of a matrix to zero.
void matrix_zero(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 0.0;
        }
    }
}

// Initialize the global matrices A, B, SEQ_MATRIX, and PAR_MATRIX.
void mmm_init() {
    A = matrix_allocate(size);
    B = matrix_allocate(size);
    SEQ_MATRIX = matrix_allocate(size);
    PAR_MATRIX = matrix_allocate(size);
    
    matrix_fill(A, size); // Fill matrices A and B with random values.
    matrix_fill(B, size);
    matrix_zero(SEQ_MATRIX, size); // Initialize SEQ_MATRIX and PAR_MATRIX with zeros.
    matrix_zero(PAR_MATRIX, size);
}

// Free the memory allocated for all matrices.
void mmm_freeup() {
    for (int i = 0; i < size; i++) {
        free(A[i]);
        free(B[i]);
        free(SEQ_MATRIX[i]);
        free(PAR_MATRIX[i]);
    }
    free(A);
    free(B);
    free(SEQ_MATRIX);
    free(PAR_MATRIX);
}

// Reset a matrix to zero before reuse.
void mmm_reset(double **matrix) {
    matrix_zero(matrix, size);
}

// Sequential matrix multiplication.
void mmm_seq(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0; // Ensure the element is initialized before accumulation.
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j]; // Multiply and accumulate.
            }
        }
    }
}

// Worker function for parallel matrix multiplication.
void *mmm_par_worker(void *arg) {
    WorkerArgs *args = (WorkerArgs *)arg;
    // Perform multiplication only for the assigned row range.
    for (int i = args->start_row; i < args->end_row; i++) {
        for (int j = 0; j < size; j++) {
            args->C[i][j] = 0; // Reset before calculation.
            for (int k = 0; k < size; k++) {
                args->C[i][j] += args->A[i][k] * args->B[k][j];
            }
        }
    }
    return NULL;
}

// Parallel matrix multiplication using pthreads.
void mmm_par(double **A, double **B, double **C, int n) {
    pthread_t threads[num_threads];
    WorkerArgs args[num_threads];
    
    // Distribute rows among threads.
    int rows_per_thread = n / num_threads;
    for (int i = 0; i < num_threads; i++) {
        args[i].start_row = i * rows_per_thread;
        args[i].end_row = (i + 1) * rows_per_thread;
        if (i == num_threads - 1) args[i].end_row = n; // Last thread may take extra rows.
        args[i].A = A;
        args[i].B = B;
        args[i].C = C;
        
        pthread_create(&threads[i], NULL, mmm_par_worker, &args[i]);
    }
    
    // Wait for all threads to complete.
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Verify the correctness of the parallel computation by comparing to SEQ_MATRIX.
double mmm_verify(double **SEQ_MATRIX, double **PAR_MATRIX, int n) {
    double max_error = 0.0;
    // Iterate over all elements and compute the maximum absolute difference.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double error = fabs(SEQ_MATRIX[i][j] - PAR_MATRIX[i][j]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    return max_error;
}

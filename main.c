#include "mmm.h"
#include "rtclock.h"
#include <stdio.h>
#include <stdlib.h>
//Author @Curtis Alves
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <S|P> <size> [num_threads]\n", argv[0]);
        return 1;
    }

    char mode = argv[1][0];
    size = atoi(argv[2]);
    num_threads = (mode == 'P' && argc == 4) ? atoi(argv[3]) : 1;

    if (size <= 0 || (mode == 'P' && num_threads <= 0)) {
        printf("Error: size and num_threads must be positive integers.\n");
        return 1;
    }

    // Initialize matrices
    mmm_init();

    double start, end, seq_time, par_time, maxError;

    // Perform sequential matrix multiplication
    if (mode == 'S' || mode == 'P') { // Also calculate sequential time for parallel mode for speedup calculation
        start = rtclock();
        for (int i = 0; i < 3; i++) {
            mmm_seq(A, B, SEQ_MATRIX, size);
            if (i < 2) mmm_reset(SEQ_MATRIX);
        }
        end = rtclock();
        seq_time = (end - start) / 3;
    }

    // Perform parallel matrix multiplication if mode is 'P'
    if (mode == 'P') {
        mmm_reset(PAR_MATRIX); // Ensure PAR_MATRIX is reset before parallel computation
        start = rtclock();
        for (int i = 0; i < 3; i++) {
            mmm_par(A, B, PAR_MATRIX, size);
            if (i < 2) mmm_reset(PAR_MATRIX);
        }
        end = rtclock();
        par_time = (end - start) / 3;
    }

    // Verify the correctness of the parallel computation against SEQ_MATRIX
    if (mode == 'P') {
        maxError = mmm_verify(SEQ_MATRIX, PAR_MATRIX, size);
    }

    // Output the results
    printf("========\n");
    printf("mode: %s\n", mode == 'S' ? "sequential" : "parallel");
    printf("thread count: %d\n", num_threads);
    printf("size: %d\n", size);
    printf("========\n");
    if (mode == 'S') {
        printf("Sequential Time (avg of 3 runs): %.6f sec\n", seq_time);
    } else if (mode == 'P') {
        printf("Sequential Time (avg of 3 runs): %.6f sec\n", seq_time);
        printf("Parallel Time (avg of 3 runs): %.6f sec\n", par_time);
        printf("Speedup: %.6f\n", seq_time / par_time);
        printf("Verifying... largest error between parallel and sequential matrix: %.6f\n", maxError);
    }

    // Free all allocated resources
    mmm_freeup();

    return 0;
}

#ifndef MMM_H_
#define MMM_H_
//author @Curtis Alves
#include <pthread.h>

extern unsigned int size, num_threads;
extern double **A, **B, **SEQ_MATRIX, **PAR_MATRIX;

void mmm_init();
void mmm_freeup();
void mmm_reset(double **matrix);
double **matrix_allocate(int n);
void matrix_fill(double **matrix, int n);
void matrix_zero(double **matrix, int n);
void mmm_seq(double **A, double **B, double **C, int n);
void* mmm_par_worker(void *arg);
void mmm_par(double **A, double **B, double **C, int n);
double mmm_verify(double **SEQ_MATRIX, double **PAR_MATRIX, int n);

#endif /* MMM_H_ */

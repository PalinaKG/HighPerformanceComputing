#ifndef _MATMULT_FUN
#define _MATMULT_FUN

/* m: no. of rows for A & C, n: no. of cols for B & C,
 * k: no. of cols for A and no. of rows for B
 * A: m x k matrix
 * B: k x n matrix
 * C: resulting matrix of matrix multiplication
*/

void matmult_nat(int m, int n, int k, double **A, double **B, double **C );

void matmult_lib(int m, int n, int k, double **A, double **B, double **C);

void matmult_mnk(int m, int n, int k, double **A, double **B, double **C );

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C );

void matmult_nkm(int m, int n, int k, double **A, double **B, double **C );

void matmult_mkn(int m, int n, int k, double **A, double **B, double **C );

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C );

void matmult_knm(int m, int n, int k, double **A, double **B, double **C);

void mat_print(int m, int n, double **A);

#endif

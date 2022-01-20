#ifndef _MATMULT_FUN
#define _MATMULT_FUN
#include <helper_cuda.h>

/* m: no. of rows for A & C, n: no. of cols for B & C,
 * k: no. of cols for A and no. of rows for B
 * A: m x k matrix
 * B: k x n matrix
 * C: resulting matrix of matrix multiplication
*/

extern "C" {
void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C);
}

extern "C" {
void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C);
}

extern "C" {
void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C);
}

extern "C" {
void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C);
}

extern "C" {
void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C);
}

extern "C" {
void matmult_lib(int m, int n, int k, double *A, double *B, double *C);
}

extern "C" {
void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C);
}


extern "C" {
void mat_print(int m, int n, double *A);
}

__global__ void gpu1_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C);

__global__ void gpu2_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C);

__global__ void gpu3_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C);

__global__ void gpu4_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C, int elements);

__global__ void gpu5_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C);

#endif

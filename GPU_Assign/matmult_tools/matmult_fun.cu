#include "matmult_func.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C" {
	#include <cblas.h>
	#include <omp.h>

	#define BLOCK_SIZE 11;
}

extern "C" {
void matmult_nat(int m, int n, int k, double *A, double *B, double *C) {
    
    int i, j, l;
    double sum;
    /*
    printf("A\n");
    mat_print(m,k,A);
    printf("B\n");
    mat_print(k,n,B);
    */
    for (l = 0; l < m; l++)
    {
        for (j = 0; j < n; j++)
        {
            sum = 0.0;
            for (i = 0; i < k; i++)
            {
                sum += A[l*k+i] * B[i*n+j]; 
            }
            C[l*n+j] = sum;
        }
    }


   /*
    printf("C\n");
    mat_print(m,n,C); 
  */ 
}
}

void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, *A, k, *B, n, 0.0, *C, n);
    
    //mat_print(m,n,C);
}


int min(int a, int b){
	
	if (a < b)
	{
		return a;
	}
	else
	{
		return b;
	} 	
}


void mat_print(int m, int n, double **A){

	for (int i = 0; i < m; i++)
    	{
        	for (int j=0; j < n; j++)
        	{
            	printf("%.2f     ", A[i][j]);
        	}
        	printf("\n");
    	}
	printf("\n\n");

}

#include "matmult_fun.h"
#include <stdio.h>
#include <cblas.h>

void matmult_nat(int m, int n, int k, double **A, double **B, double **C) {
    
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
                sum += A[l][i] * B[i][j]; 
            }
            C[l][j] = sum;
        }
    }


   /*
    printf("C\n");
    mat_print(m,n,C); 
   */
}

void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, *A, k, *B, n, 0.0, *C, n);
    
    //mat_print(m,n,C);
}



void matmult_mnk(int m, int n, int k, double **A, double **B, double **C) {
	matmult_nat(m,n,k,A,B,C);
}


void matmult_nmk(int m, int n, int k, double **A, double **B, double **C){
	
   	int i, j, l;
   	double sum;
    
   	printf("A\n");
   	mat_print(m,k,A);
   	printf("B\n");
	mat_print(k,n,B);
	
	for (l = 0; l < n; l++)
	{	
	for (j = 0; j < m; j++)
        {
            sum = 0.0;
            for (i = 0; i < k; i++)
            {
                sum += A[j][i] * B[i][l]; 
            }
            C[j][l] = sum;
        }
    }




   
    printf("C\n");
    mat_print(m,n,C); 
   


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

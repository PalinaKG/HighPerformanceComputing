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
    /*
   	printf("A\n");
   	mat_print(m,k,A);
   	printf("B\n");
	mat_print(k,n,B);
	*/

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



	/*
    printf("C\n");
    mat_print(m,n,C); 
   	*/


}


void matmult_nkm(int m, int n, int k, double **A, double **B, double **C){
	
   	int i, j, l;
   	double sum;
    /*
   	printf("A\n");
   	mat_print(m,k,A);
   	printf("B\n");
	mat_print(k,n,B);
	*/
	
	for (int i = 0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
				C[i][j] = 0.0;
        }
    }

	for (l = 0; l < n; l++)
	{	
	for (j = 0; j < k; j++)
        {
            for (i = 0; i < m; i++)
            {
                C[i][l] +=  A[i][j] * B[j][l]; 
            }
        }
    }
	/*	
    printf("C\n");
    mat_print(m,n,C); 
   	*/
}

void matmult_mkn(int m, int n, int k, double **A, double **B, double **C){
	
   	int i, j, l;
   	double sum;
    /*
   	printf("A\n");
   	mat_print(m,k,A);
   	printf("B\n");
	mat_print(k,n,B);
	*/

	
	for (int i = 0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
				C[i][j] = 0.0;
        }
    }

	for (l = 0; l < m; l++)
	{	
	for (j = 0; j < k; j++)
        {
            for (i = 0; i < n; i++)
            {
                C[l][i] +=  A[l][j] * B[j][i]; 
            }
        }
    }
	/*
    printf("C\n");
    mat_print(m,n,C); 
   	*/
}

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C){
	
   	int i, j, l;
   	double sum;
    /*
   	printf("A\n");
   	mat_print(m,k,A);
   	printf("B\n");
	mat_print(k,n,B);
	*/
	
	for (int i = 0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
				C[i][j] = 0.0;
        }
    }

	for (l = 0; l < k; l++)
	{	
	for (j = 0; j < m; j++)
        {
            for (i = 0; i < n; i++)
            {
                C[j][i] +=  A[j][l] * B[l][i]; 
            }
        }
    }
	/*	
    printf("C\n");
    mat_print(m,n,C); 
   	*/
}

void matmult_knm(int m, int n, int k, double **A, double **B, double **C){
	
   	int i, j, l;
   	double sum;
    /*
   	printf("A\n");
   	mat_print(m,k,A);
   	printf("B\n");
	mat_print(k,n,B);
	*/
	
	for (int i = 0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
				C[i][j] = 0.0;
        }
    }

	for (l = 0; l < k; l++)
	{	
	for (j = 0; j < n; j++)
        {
            for (i = 0; i < m; i++)
            {
                C[i][j] +=  A[i][l] * B[l][j]; 
            }
        }
    }
	/*	
    printf("C\n");
    mat_print(m,n,C); 
   	*/
}


void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs){
    int i, j, l, ii, jj, ll, l_min, j_min, i_min;
    double sum;
   /* 
    printf("A\n");
    mat_print(m,k,A);
    printf("B\n");
    mat_print(k,n,B);
    */
	for (int i = 0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
				C[i][j] = 0.0;
        }
    }
    

	for (l = 0; l < m; l+=bs)
    {
		l_min = min(bs,m-l);
        for (j = 0; j < k; j+=bs)
        {
			j_min = min(bs,k-j);
            for (i = 0; i < n; i+=bs)
            {
				i_min = min(bs,n-i);
            	for (ll = 0; ll < l_min; ll++)
            	{
					for (jj = 0; jj < j_min; jj++)
					{
                		for (ii = 0; ii < i_min; ii++)
						{
							C[ll+l][ii+i] += A[ll+l][jj+j] * B[jj+j][ii+i];
						}            
					}
            	}	
        	}	
    	}
	}
/*
    printf("C\n");
    mat_print(m,n,C);  
*/
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

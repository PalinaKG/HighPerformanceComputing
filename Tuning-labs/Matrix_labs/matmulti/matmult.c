#include <stdio.h>
void
matmult(int m, int k, int n, double **A, double **B, double **C) {
    
    int i, j, kk;
    double sum;

    for (kk = 0; kk < m; kk++)
    {
        for (j = 0; j < n; j++)
        {
            sum = 0;
            for (i = 0; i < k; i++)
            {
                
                sum += A[i][kk] * B[j][i];
                
            }
            //printf("HELLO: %f", B[j][i]);
            C[j][kk] = sum;
        }
    }
}


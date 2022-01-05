void
matmult(int m, int k, int n, double **A, double **B, double **C) {
    
    int i, j, kk, sum;

    for(i = 0; i < m; i++)
	for(j = 0; j < n; j++)
	    C[i][j] = A[i][j]*B[i][j];
    
    
    for (kk = 0, kk < m; kk++)
    {
        for (j = 0; j < n; n++)
        {
            sum = 0
            for (i = 0; i < k; i++)
            {
                
                sum += A[i][kk] * B[j][i]
            }
            C[kk][j] = sum
        }
    }
}


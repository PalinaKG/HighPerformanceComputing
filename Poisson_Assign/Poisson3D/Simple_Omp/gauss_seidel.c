#include <math.h>
#include <string.h>
#include <stdio.h>

int gauss_seidel(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    float d = 1.0/0.0;
    int counter = 0;
	int n = N+2;
    double delta = (1.0/(double)N)*(1.0/(double)N);
	// var for calc frobenius norm
	double sum, val=0.0, tmp=0.0;
    
    while (d > threshold && counter < k_max) {
    	sum = 0.0;
		// update u
        #pragma omp parallel for ordered(2) private(j,k)
		for (int i = 1; i < (N + 1); i++)
    	{
        	for (int j = 1; j < (N + 1); j++)
        	{
                #pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1)
            	for (int k = 1; k < (N + 1); k++)
            	{
					tmp = u[i][j][k];		
                	u[i][j][k] = (1.0/6.0)*(u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + delta*f[i][j][k]);
            		val = u[i][j][k] - tmp;
					sum += val*val;				
				}
                #pragma omp ordered depend(source)
        	}
    	}
        d = sqrt(sum); 
      	printf("%f\n",d);
		counter = counter + 1;
    }
    return counter;
}


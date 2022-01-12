/* gauss_Seidel.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>

void update(int N, double ***f, double ***u);
double frobenius_norm(double ***u, double ***u_old, int N);

int gauss_seidel(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    float d = 1.0/0.0;
    int k = 0;
	int n = N+2;
    
    while (d > threshold && k < k_max) {
        memcpy(&u_old[0][0][0],&u[0][0][0],n*n*n*sizeof(&u[0][0][0]));
        update(N, f, u);
        d = frobenius_norm(u, u_old, N);
        k = k + 1;
    }
    return k;
}


void update(int N, double ***f, double ***u)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;

#pragma omp parallel for ordered(2) private(j,k)
    for (i = 1; i < (N + 1); i++)
    {
        for (j = 1; j < (N + 1); j++)
        {
#pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1)
            for (k = 1; k < (N + 1); k++)
            {
                u[i][j][k] = (1.0/6.0)*(u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + delta*f[i][j][k]);
            }
#pragma omp ordered depend(source)
        }
    }
}

double frobenius_norm(double ***u, double ***u_old, int N)
{
    double sum = 0.0;
    double value = 0.0;
    for (int i = 0; i < (N + 2); i++)
    {
        for (int j = 0; j < (N + 2); j++)
        {
            for (int k = 0; k < (N + 2); k++)
            {
                value = u[i][j][k] - u_old[i][j][k];
                sum += (value * value);
            }
        }
    }
    return sqrt(sum);
}

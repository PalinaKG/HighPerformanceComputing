/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>

void update(int N, double ***f, double ***u, double ***u_old);
double frobenius_norm(double ***u, double ***u_old, int N);

int jacobi(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    float d = 1.0/0.0;
    int counter = 0;
	int n = N+2;
    double ***temp_uold;
    
    while (d > threshold && counter < k_max) {
        // memcpy(&u_old[0][0][0],&u[0][0][0],n*n*n*sizeof(&u[0][0][0]));
        temp_uold = u_old;
        u_old=u;
        u = temp_uold;
        update(N, f, u, u_old);
        d = frobenius_norm(u, u_old, N);
        counter = counter + 1;
        printf("%d", counter);
        printf("%f", d);
    }
    
    return counter;
}


void update(int N, double ***f, double ***u, double ***u_old)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
    

    #pragma omp parallel default(none) shared(u,u_old, N) \
    private(i,j,k) firstprivate(f, delta)
    {
    #pragma omp for schedule(dynamic, 100)    
    for (i = 1; i < (N + 1); i++)
    {
        for (j = 1; j < (N + 1); j++)
        {
            for (k = 1; k < (N + 1); k++)
            {
                u[i][j][k] = (1.0/6.0)*(u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta*f[i][j][k]);
            }
        }
    }
    } //end of parallel
}

double frobenius_norm(double ***u, double ***u_old, int N)
{
    double sum = 0.0;
    double value = 0.0;
    for (int i1 = 0; i1 < (N + 2); i1++)
    {
        for (int j1 = 0; j1 < (N + 2); j1++)
        {
            for (int k1 = 0; k1 < (N + 2); k1++)
            {
                value = u[i1][j1][k1] - u_old[i1][j1][k1];
                sum += (value * value);
            }
        }
    }
    return sqrt(sum);
}

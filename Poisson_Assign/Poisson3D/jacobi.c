/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>

void update(int N, double ***f, double ***u, double ***u_old);
double frobenius_norm(double ***u, double ***u_old, int N);

void
jacobi(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    float d = 1.0/0.0;
    int k = 0;
    
    while (d > threshold && k < k_max) {
        memcpy(&u_old[0][0][0],&u[0][0][0],N*N*N*sizeof(&u[0][0][0]));
        update(N, f, u, u_old);
        d = frobenius_norm(u, u_old, N);
        k = k + 1;
    }
    
}


void update(int N, double ***f, double ***u, double ***u_old)
{
    int delta = 4 / ((N-1)*(N-1));
    for (int i = 1; i < (N + 1); i++)
    {
        for (int j = 1; j < (N + 1); j++)
        {
            for (int k = 1; k < (N + 1); k++)
            {
                u[i][j][k] = (1/6)*(u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta*f[i][j][k]);
            }
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
                sum += value * value;
            }
        }
    }
    return sqrt(sum);
}
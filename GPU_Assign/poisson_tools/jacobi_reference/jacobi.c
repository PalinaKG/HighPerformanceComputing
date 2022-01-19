/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>

void update(int N, double ***f, double ***u, double ***u_old);



int jacobi(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    
	int n = N+2;
    double ***temp_uold;

    memcpy(&u_old[0][0][0],&u[0][0][0],n*n*n*sizeof(&u[0][0][0]));
    
    for (int i = 0; i < k_max; i++) {
        update(N, f, u, u_old);
        temp_uold = u_old;
        u_old=u;
        u = temp_uold;
    }
    return k_max;

    
}


void update(int N, double ***f, double ***u, double ***u_old)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
 	double s = 1.0/6.0;  

    #pragma omp parallel for default(none) shared(u,u_old, N) \
    private(i,j,k) firstprivate(f, delta,s) schedule(dynamic,3)
    for (i = 1; i < (N + 1); i++)
    {
        for (j = 1; j < (N + 1); j++)
        {
            for (k = 1; k < (N + 1); k++)
            {
                u[i][j][k]= s*(u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] \
                + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] \
                + delta*f[i][j][k]); 
            }

        }
    }
}



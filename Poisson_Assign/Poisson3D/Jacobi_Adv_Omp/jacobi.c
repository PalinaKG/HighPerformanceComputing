/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>



int jacobi(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    float d = 1.0/0.0;
    // int counter = 0;
	int n = N+2;
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k,i2,j2,k2;
    double sum = 0.0;
    double value = 0.0;
    
    # pragma omp parallel default(none) shared(u,u_old,d ,n,delta,N) \
    firstprivate(f,k_max) private(i,j,k,i2,j2,k2,value)
    for(int counter = 0; counter < k_max;counter++) {
        memcpy(&u_old[0][0][0],&u[0][0][0],n*n*n*sizeof(&u[0][0][0]));
    // # pragma omp parallel for default(none) shared(u,u_old,f,N,delta) 
    // UPDATE
    # pragma omp for 
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
        //FROBNORM
    //  #pragma omp for 
    //     for (i2 = 0; i2 < (N + 2); i2++)
    //     {
    //         for (j2 = 0; j2 < (N + 2); j2++)
    //         {
    //             for (int k2 = 0; k2 < (N + 2); k2++)
    //             {
    //                 value = u[i][j][k] - u_old[i][j][k];
    //                 sum += (value * value);
    //             }
    //         }
    //     }
    // #pragma omp critical    
    // d = sqrt(sum);
         
    }

    return k_max;
}




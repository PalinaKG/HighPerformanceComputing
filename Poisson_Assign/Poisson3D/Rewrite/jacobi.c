/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>

double update(int N, double ***f, double ***u, double ***u_old);



int jacobi(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold) {
    
    float d = 1.0/0.0;
    int counter = 0;
	int n = N+2;

    
    while (d > threshold && counter < k_max) {
        memcpy(&u_old[0][0][0],&u[0][0][0],n*n*n*sizeof(&u[0][0][0]));
        d = update(N, f, u, u_old);
        counter = counter + 1;
	    //printf("%d ", counter);
        //printf("%.3f \n", d);
    }
    return counter;

    
}


double update(int N, double ***f, double ***u, double ***u_old)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    double sum = 0.0;
    double value = 0.0;
    double temp_unew = 0.0;

    
    for (int i = 1; i < (N + 1); i++)
    {
        for (int j = 1; j < (N + 1); j++)
        {
            for (int k = 1; k < (N + 1); k++)
            {
                u[i][j][k]= (1.0/6.0)*(u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] \
                + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] \
                + delta*f[i][j][k]);
                value = u[i][j][k] - u_old[i][j][k];
                sum += (value * value);
                      
            }

        }
    }
    return sqrt(sum);
}



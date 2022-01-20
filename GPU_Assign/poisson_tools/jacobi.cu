/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"

void update(int N, double ***f, double ***u, double ***u_old);
void jacobi(double ***f, double ***u, double ***u_old, int N, int k_max);
void jacobi_seq(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max);
__global__ void kernel_seq(int N, double ***f, double ***u, double ***u_old);


//****************PART 0 - REFERENCE COMPARISON VERSION*******************
// reference version from assignment 2
// threshold has been removed as well as norm calculations
// that it - stop criteria removed for simplicity
void jacobi(double ***f, double ***u, double ***u_old, int N, int k_max) {
    
	int n = N+2;
    double ***temp_uold;

    memcpy(&u_old[0][0][0],&u[0][0][0],n*n*n*sizeof(&u[0][0][0]));
    
    for (int counter = 0; counter < k_max; counter++) {
        update(N, f, u, u_old);
        temp_uold = u_old;
        u_old=u;
        u = temp_uold;
    }
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

//****************PART 0 - END OF REFERENCE VERSION FOR COMPARISON*******************


//****************PART 1 - sTART OF Sequential GPU version (baseline)  *******************
void jacobi_seq(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max){
    

    //declare u, u_old and f for GPU side
    double 	***u_d = NULL;
    double  ***u_old_d = NULL;
    double  ***f_d = NULL;

    // Allocate 3x 3d array in device memory. (GPU side)
    if ( (u_d = d_malloc_3d_gpu(N + 2, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

        // Allocate 3x 3d array in device memory. (GPU side)
    if ( (u_old_d = d_malloc_3d_gpu(N + 2, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

        // Allocate 3x 3d array in device memory. (GPU side)
    if ( (f_d = d_malloc_3d_gpu(N + 2, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

    // do a CPU → GPU transfer of u and f for the initialized data
    transfer_3d(u_d, u_h, N + 2, N + 2, N + 2, cudaMemcpyHostToDevice);
    transfer_3d(f_d, f_h, N + 2, N + 2, N + 2, cudaMemcpyHostToDevice);
    transfer_3d(u_old_d, u_old_h, N + 2, N + 2, N + 2, cudaMemcpyHostToDevice);

    double ***temp_uold;
    // Launch your Jacobi iteration kernel inside a CPU controlled iteration loop to get
    // global synchronization between each iteration step
   
    for (int counter = 0; counter < iter_max; counter++) {
        temp_uold = u_old_d;
        u_old_d=u_d;
        u_d = temp_uold;
        kernel_seq<<<1,1>>>(N, f_d, u_d, u_old_d);
    }

    // When all iterations are done, transfer the result from GPU → CPU
    transfer_3d(u_h, u_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d(f_h, f_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d(u_old_h, u_old_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);

    free_gpu(u_d);
    free_gpu(u_old_d);
    free_gpu(f_d);
}

__global__ void kernel_seq(int N, double ***f, double ***u, double ***u_old)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
 	double s = 1.0/6.0;  

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


//****************PART 1 - END OF Sequential GPU version (baseline)  *******************


//****************PART 2 - START OF  Naive GPU version – one thread per element  *******************
//****************PART 2 - END OF  Naive GPU version – one thread per element  *******************

//****************PART 3 - START OF  Multi-GPU version  *******************
//****************PART 3 - END OF  Multi-GPU version  *******************

//****************PART 4 - START OF  naive + stop version  *******************
//****************PART 4 - END OF  naive + stop version  *******************

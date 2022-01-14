/* main.c - Poisson problem in 3D andri test commit
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <math.h>
#include <time.h>
#include <omp.h>

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100
void Initialize_U(double ***u, int N, int start_t);
void Initialize_F(double ***f, int N);

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {
    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    char    *fun_type = "";
	double 	***u = NULL;
    double  ***u_old = NULL;
    double  ***f = NULL;
    double start; 
	double end; 
    int iter;
    clock_t start_t, end_t;
    double total_time;


    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }
    // allocate memory
    if ( (u = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    
    if ( (f = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    
    if ( (u_old = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    
	Initialize_F(f,N);
    Initialize_U(u, N,start_T);
    

	//x3 because we have 3 matrices, u, uold and f
    float mem = sizeof(double) * (N+2) * (N+2) * (N+2) * 3;


    start_t = mytimer();
    #ifdef _JACOBI
        //the interations are dynamic, we should return the num of iteration from jacobi
        iter = jacobi(f, u, u_old, N, iter_max, tolerance);
		fun_type = "j";
    #endif
    #ifdef _GAUSS_SEIDEL
		iter = gauss_seidel(f, u, u_old, N, iter_max, tolerance);
		fun_type = "gs";
    #endif
    end_t = mytimer();
    // 8 floating point operations in the jakobi update
    double flops = 8 * N * N * N * (double)iter / 10e6;


    //total time
    total_time = delta_t(start_t, end_t) / 1000;

    //flops per second
    //double flopSec = flops/total_time;

    printf("%.3f ", flops); //total Mflops
    printf("%.3f ", mem/1024.0); //memory in kbytes
    printf("%8.3f ", total_time); //total time in sec
    printf("%d ", N); //grid size
    printf("%d ", iter); //number of iterations in jacobi
    printf("%.3f\n", flops/total_time); //flops/s
    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, fun_type, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, u);
	    break;
	case 4:
	    output_ext = ".vtk";		
	    sprintf(output_filename, "%s_%s_%d%s", output_prefix, fun_type, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: \n", output_filename);
	    print_vtk(output_filename, N+2, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);
    free(u_old);
    free(f);

    return(0);
}


void Initialize_U(double ***u, int N, int start_T)
{
    for (int i = 0; i < (N + 2); i++)
    {
        for (int j = 0; j < (N + 2); j++)
        {
            for (int k = 0; k < (N + 2); k++)
            {
                u[i][j][k] = start_T;
            }
        }
    }
    
	
	for (int i = 0; i < (N + 2); i++)
    {
        for (int k = 0; k < (N + 2); k++)
        {
            u[i][0][k] = 0.0;
            u[i][N+1][k] = 20.0;
            u[0][i][k] = 20.0;
            u[N+1][i][k] = 20.0;
            u[i][k][0] = 20.0;
            u[i][k][N+1] = 20.0;
        }
    }
}

void Initialize_F(double ***f, int N)
{
    for (int i = 0; i < (N + 2); i++)
    {
        for (int j = 0; j < (N + 2); j++)
        {
            for (int k = 0; k < (N + 2); k++)
            {
                f[i][j][k] = 0.0;
            }
        }
    }
    
    int x1 = 0;
    int x2 =floor( (N*5.0/16.0) ); //(1 + (-3/8)) * 1/2 
    int y1 = 0;
    int y2 = floor((1/4.0)*N);
    int z1 = ceil((1/6.0)*N);
    int z2 = floor((1/2.0)*N);

    for (int i = x1; i <= x2; i++)
    {
        for (int j = y1; j <= y2; j++)
        {
            for (int k = z1; k <= z2; k++)
            {
                f[i][j][k] = 200.0;
            }
        }
    }
    
}

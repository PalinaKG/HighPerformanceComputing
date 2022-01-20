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
#include "jacobi.h"

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
    double 	***u_h = NULL;
    double  ***u_old_h = NULL;
    double  ***f_h = NULL;
    double start; 
	double end; 
    int iter = 10;
    double start_t, end_t;
    double total_time;
    int func_type = 1;

    const long nElms = N * N * N; // Number of elements.


    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }
    if(argc == 7){
        func_type = atoi(argv[6]); //function type
    }

    // Allocate 3x 3d array in host memory.
    if ( (u_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u_h: allocation failed");
        exit(-1);
    }
    
    if ( (f_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f_h: allocation failed");
        exit(-1);
    }
    
    if ( (u_old_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u_old_h: allocation failed");
        exit(-1);
    }

    start_t = omp_get_wtime();

    // let the CPU initialize vectors u and f
    Initialize_F(f_h,N);
    Initialize_U(u_h, N,start_T);
    

	//x3 because we have 3 matrices, u, uold and f
    float mem = sizeof(double) * (N+2) * (N+2) * (N+2) * 3;


    //the iterations are static and we're always returning the iter_max value
    if(func_type == 0){
        jacobi(f_h, u_h, u_old_h, N, iter_max);
    }
    else if(func_type == 1){
        jacobi_seq(f_h, u_h, u_old_h, N, iter_max);
    }
    else if(func_type == 3){
        jacobi_multi(f_h, u_h, u_old_h, N, iter_max);
    }
    else if(func_type == 4){
        jacobi_stop(f_h, u_h, u_old_h, N, iter_max,tolerance);
    }
    iter = iter_max;

    end_t = omp_get_wtime();

    

    // 8 floating point operations in the jakobi update
    double flops = 8.0 * (N+2) * (N+2) * (N+2) * (double)iter / 1e6;

    //total time
    //total_time = delta_t(start_t, end_t) / 1000;
    // printf("%8.3f", total_time);
    total_time = (end_t - start_t);

   
    /* Print n and results  */
	
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
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, u_h);
        break;
    case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, u_h);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u_h);
    free(u_old_h);
    free(f_h);

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

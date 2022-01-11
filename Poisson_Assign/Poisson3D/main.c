/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <math.h>

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100
void boundaries(double ***u, int N);
void initialize_f(double ***f, int N);

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
    double 	***u = NULL;
    double  ***u_old = NULL;
    double  ***f = NULL;


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
    
    
    initialize_f(f,N);
    
    
    
    //XYZ
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
   
    boundaries(u, N);

    #ifdef _JACOBI
        jacobi(f, u, u_old, N, iter_max, tolerance);
    #endif
    

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);

    return(0);
}

void boundaries(double ***u, int N)
{
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

void initialize_f(double ***f, int N)
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
    int x2 = floor((5/16)*N); //(1 + (-3/8)) * 1/2
    int y1 = 0;
    int y2 = floor((1/4)*N);
    int z1 = ceil((1/6)*N);
    int z2 = floor((1/2)*N);
    
    
    for (int i = x1; i < x2; i++)
    {
        for (int j = y1; j < y2; j++)
        {
            for (int k = z1; k < z2; k++)
            {
                f[i][j][k] = 200.0;
            }
        }
    }
    
}

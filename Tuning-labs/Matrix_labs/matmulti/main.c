#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"		/* helper functions	        */
#include "matmult.h"		/* my matrix add fucntion	*/

#define NREPEAT 1		/* repeat count for the experiment loop */

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {
    printf("HELLO");

    int    i, m, n, k, N = NREPEAT;
    double **A, **B, **C;
    double tcpu1; 

    clock_t t1, t2;

    for (m = 200; m <= 3500; m += 300) {
    //for (m = 5; m <= 5; m += 5) {
        
	n = m + 25;
        //n=10;
    k = 5;
        

	/* Allocate memory */
        
	A = malloc_2d(k, m);
	B = malloc_2d(n, k);
	C = malloc_2d(n, m);
	if (A == NULL || B == NULL | C == NULL) {
	    fprintf(stderr, "Memory allocation error...\n");
	    exit(EXIT_FAILURE);
	}

	/* initialize with useful data - last argument is reference */
	init_data(k,m,A,1.0);
    init_data(n,k,B,2.0);

//	 timings for matadd
	t1 = mytimer();
	for (i = 0; i < N; i++)
	    matmult(m, k, n, A, B, C);
	t2 = mytimer();
	tcpu1 = delta_t(t1, t2) / N;

	//check_results("main", m, n, C);

	/* Print n and results  */
	printf("%4d %4d %8.3f\n", m, n, tcpu1);

    for (int a = 0; a < m; a++)
    {
        for (int b=0; b < k; b++)
        {
            printf("%.2f     ", A[b][a]);
        }
        printf("\n");
    }
        printf("\n\n\n");
        
        for (int a = 0; a < k; a++)
        {
            for (int b=0; b < n; b++)
            {
                printf("%.2f     ", B[b][a]);
            }
            printf("\n");
        }
        
        printf("\n\n\n");
        for (int a = 0; a < m; a++)
        {
            for (int b=0; b < n; b++)
            {
                printf("%.2f     ", C[b][a]);
            }
            printf("\n");
        }
    
	/* Free memory */
	free_2d(A);
	free_2d(B);
	free_2d(C);
    }

    return EXIT_SUCCESS;
}

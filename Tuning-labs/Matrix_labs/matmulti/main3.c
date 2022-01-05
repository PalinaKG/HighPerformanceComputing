#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"        /* helper functions            */
#include "matmult.h"        /* my matrix add fucntion    */

#define NREPEAT 1        /* repeat count for the experiment loop */

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {

    int    i, m, n, k, N = NREPEAT;
    double **A, **B, **C;
    double tcpu1;

    clock_t t1, t2;

    //for (m = 200; m <= 3500; m += 300) {
    //for (m = 5; m <= 10; m += 5) {
    n = 4;
    k = 5;
    double counter = 0;
    for (int b = 0; b < m; b++)
    {
        for (int c = 0; c < n; c++)
        {
            A[b][c] = counter;
            B[b][c] = counter;
            C[b][c] = counter;
            counter++;
        }
    }
    /* Allocate memory */
    

    /* initialize with useful data - last argument is reference */

    /* timings for matadd */
    t1 = mytimer();
    for (i = 0; i < N; i++)
        matmult(m, k, n, A, B, C);
//    t2 = mytimer();
//    tcpu1 = delta_t(t1, t2) / N;

    //check_results("main", m, n, C);

    /* Print n and results  */
//    printf("%4d %4d %8.3f\n", m, n, tcpu1);
//
//    for (int a = 0; a < n; a++)
//    {
//        for (int b=0; b < m; b++)
//        {
//            printf("%4f", C[b][a]);
//        }
//        printf("\n");
//    }
//

   // }

    return EXIT_SUCCESS;
}


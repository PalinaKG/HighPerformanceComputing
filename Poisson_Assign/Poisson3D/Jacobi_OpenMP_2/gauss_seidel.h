/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

// define your function prototype here
int gauss_seidel(double ***f, double ***u, double ***u_old, int N, int k_max, double threshold);

#endif

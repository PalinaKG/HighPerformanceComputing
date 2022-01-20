/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

void update(int N, double ***f, double ***u, double ***u_old);
void jacobi(double ***f_h, double ***u_h, double ***u_old_h, int N, int k_max);
void jacobi_seq(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max);
void jacobi_naive(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max);
__global__ void kernel_seq(int N, double ***f, double ***u, double ***u_old);
__global__ void kernel_naive(int N, double ***f, double ***u, double ***u_old);

#endif

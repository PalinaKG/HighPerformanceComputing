/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include <helper_cuda.h>


void update(int N, double ***f, double ***u, double ***u_old);
void jacobi(double ***f_h, double ***u_h, double ***u_old_h, int N, int k_max);
void jacobi_seq(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max);
void jacobi_naive(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max);
void jacobi_multi(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max);
void jacobi_stop(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max, double threshold);


__global__ void kernel_seq(int N, double ***f, double ***u, double ***u_old);
__global__ void kernel_naive(int N, double ***f, double ***u, double ***u_old);
__global__ void kernel_gpu0(int N, double ***f, double ***u, double ***u_old, double ***u_old_d1, int boundary);
__global__ void kernel_gpu1(int N, double ***f, double ***u, double ***u_old, double ***u_old_d0, int boundary);

__global__ void kernel_stop(int N, double ***f, double ***u, double ***u_old, double *d);

__inline__ __device__ double SumReduce(double val);

__inline__ __device__ double WarpReduction(double value);


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
    transfer_3d(u_old_d, u_h, N + 2, N + 2, N + 2, cudaMemcpyHostToDevice);

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

void jacobi_naive(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max){
    

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
    transfer_3d(u_old_d, u_h, N + 2, N + 2, N + 2, cudaMemcpyHostToDevice);
    // Total number of threads 2048
    //Number of threads per block 1024
    double ***temp_uold;
    // Launch your Jacobi iteration kernel inside a CPU controlled iteration loop to get
    // global synchronization between each iteration step

    dim3 num_blocks = dim3(ceil(N/32), ceil(N/8), ceil(N));
    dim3 threads_per_block = dim3(32,8,1);
    
    for (int i = 0; i < iter_max; i++) {
        temp_uold = u_old_d;
        u_old_d=u_d;
        u_d = temp_uold;
        kernel_naive<<<num_blocks,threads_per_block>>>(N, f_d, u_d, u_old_d);
        cudaDeviceSynchronize();
    }

    // When all iterations are done, transfer the result from GPU → CPU
    transfer_3d(u_h, u_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d(f_h, f_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d(u_old_h, u_old_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);

    free_gpu(u_d);
    free_gpu(u_old_d);
    free_gpu(f_d);
}

__global__ void kernel_naive(int N, double ***f, double ***u, double ***u_old)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
    double s = 1.0/6.0;  
     
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i>0 && j>0 && k>0 && j<(N+1) && k<(N+1) && i<(N+1)){
    u[i][j][k]= s*(u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] \
    + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] \
    + delta*f[i][j][k]); 
    }

        
    
}

//****************PART 2 - END OF  Naive GPU version – one thread per element  *******************



//****************PART 3 - START OF  Multi-GPU version  *******************
void jacobi_multi(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max){
    

    //declare u, u_old and f for GPU side
    double 	***u_d0 = NULL;
    double  ***u_old_d0 = NULL;
    double  ***f_d0 = NULL;
    double 	***u_d1 = NULL;
    double  ***u_old_d1 = NULL;
    double  ***f_d1 = NULL;

    int half = (N + 2) / 2;
    int nElems = (N + 2) * (N + 2) * (N + 2);

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
    // Allocate 3x 3d array in device0 memory. (GPU side)
    if ( (u_d0 = d_malloc_3d_gpu(half, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

    if ( (u_old_d0 = d_malloc_3d_gpu(half, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

    if ( (f_d0 = d_malloc_3d_gpu(half, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

    // do a CPU → GPU transfer of u and f for the initialized data
    transfer_3d_from_1d(u_d0, u_h[0][0], half, N + 2, N + 2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(f_d0, f_h[0][0], half, N + 2, N + 2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(u_old_d0, u_h[0][0], half, N + 2, N + 2, cudaMemcpyHostToDevice);


    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)

    // Allocate 3x 3d array in device1 memory. (GPU side)
    if ( (u_d1 = d_malloc_3d_gpu(half, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

        // Allocate 3x 3d array in device memory. (GPU side)
    if ( (u_old_d1 = d_malloc_3d_gpu(half, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

        // Allocate 3x 3d array in device memory. (GPU side)
    if ( (f_d1 = d_malloc_3d_gpu(half, N + 2, N + 2)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

    // do a CPU → GPU transfer of u and f for the initialized data
    transfer_3d_from_1d(u_d0, u_h[0][0] + nElems/2 , half, N + 2, N + 2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(f_d0, f_h[0][0] + nElems/2, half, N + 2, N + 2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(u_old_d0, u_h[0][0] + nElems/2, half, N + 2, N + 2, cudaMemcpyHostToDevice);

    // Total number of threads 2048
    //Number of threads per block 1024
    double ***temp_uold0;
    double ***temp_uold1;
    // Launch your Jacobi iteration kernel inside a CPU controlled iteration loop to get
    // global synchronization between each iteration step
    double dimtemp = ceil((double)N/8);
    dim3 num_blocks = dim3(dimtemp,dimtemp,ceil((double)half/8));
    dim3 threads_per_block = dim3(8,8,8);
    
    for (int i = 0; i < iter_max; i++) {

        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
        temp_uold0 = u_old_d0;
        u_old_d0=u_d0;
        u_d0 = temp_uold0;
        kernel_gpu0<<<num_blocks,threads_per_block>>>(N, f_d0, u_d0, u_old_d0, u_old_d1, floor((N+1)/2));

        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)
        temp_uold1 = u_old_d1;
        u_old_d1=u_d1;
        u_d1= temp_uold1;
        kernel_gpu1<<<num_blocks,threads_per_block>>>(N, f_d1, u_d1, u_old_d1, u_old_d0, floor((N+1)/2));
        
        checkCudaErrors(cudaDeviceSynchronize());
        cudaSetDevice(0);
        checkCudaErrors(cudaDeviceSynchronize());
    }


    // When all iterations are done, transfer the result from GPU → CPU
    transfer_3d_to_1d(u_h[0][0], u_d0, half, N + 2, N + 2, cudaMemcpyDeviceToHost);
    //transfer_3d_from_1d(f_h, f_d0[0][0], half, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d_to_1d(u_old_h[0][0], u_old_d0, half, N + 2, N + 2, cudaMemcpyDeviceToHost);

    transfer_3d_to_1d(u_h[0][0] + nElems/2, u_d1, half, N + 2, N + 2, cudaMemcpyDeviceToHost);
    //transfer_3d_from_1d(f_h, f_d1[0][0] + nElems/2, half, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d_to_1d(u_old_h[0][0]  + nElems/2, u_old_d1, half, N + 2, N + 2, cudaMemcpyDeviceToHost);

    free_gpu(u_d0);
    free_gpu(u_old_d0);
    free_gpu(f_d0);

    free_gpu(u_d1);
    free_gpu(u_old_d1);
    free_gpu(f_d1);
}

__global__ void kernel_gpu0(int N, double ***f, double ***u, double ***u_old, double ***u_old_d1, int boundary)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
    double s = 1.0/6.0;  
     
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i>0 && j>0 && k>0 && j<(N+1) && k<(N+1) && i < ((N+2)/2)){
        u[i][j][k]= u_old[i-1][j][k] + u_old[i][j-1][k] \
        + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] \
        + delta*f[i][j][k]; 

        if(i == (N/2)){
            u[i][j][k] += u_old_d1[0][j][k];
        }
        else{
            u[i][j][k] += u_old[i+1][j][k];
        }

        u[i][j][k] = s * u[i][j][k];
    }     
    
}

__global__ void kernel_gpu1(int N, double ***f, double ***u, double ***u_old, double ***u_old_d0, int boundary)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
    double s = 1.0/6.0;  
     
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;


    if(i>= 0 && j>0 && k>0 && j<(N+1) && k<(N+1) && i < ((N+2)/2)-1){
        u[i][j][k]= u_old[i+1][j][k] + u_old[i][j-1][k] \
        + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] \
        + delta*f[i][j][k]; 

        if(i == 0){
            u[i][j][k] += u_old_d0[(N/2)][j][k];
        }
        else{
            u[i][j][k] += u_old[i-1][j][k];
        }

        u[i][j][k] = s * u[i][j][k];
    } 
}   
    
//****************PART 3 - END OF  Multi-GPU version  *******************

//****************PART 4 - START OF  naive + stop version  *******************

void jacobi_stop(double ***f_h, double ***u_h, double ***u_old_h, int N, int iter_max, double threshold){
    
    //declare u, u_old and f for GPU side
    double 	***u_d = NULL;
    double  ***u_old_d = NULL;
    double  ***f_d = NULL;
	double *d;
	double myvar = 1.0/0.0;
	int counter=0;

	*d = myvar;
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
    transfer_3d(u_old_d, u_h, N + 2, N + 2, N + 2, cudaMemcpyHostToDevice);
    // Total number of threads 2048
    //Number of threads per block 1024
    double ***temp_uold;
    // Launch your Jacobi iteration kernel inside a CPU controlled iteration loop to get
    // global synchronization between each iteration step
    double dimtemp = ceil((double)N/8);
    dim3 num_blocks = dim3(dimtemp,dimtemp,dimtemp);
    dim3 threads_per_block = dim3(8,8,8);
/*
	while (*d > threshold && counter < iter_max) 
    {
        temp_uold = u_old_d;
        u_old_d=u_d;
        u_d = temp_uold;
        //kernel_stop<<<num_blocks,threads_per_block>>>(N, f_d, u_d, u_old_d,d);
        cudaDeviceSynchronize();
		
		counter = counter + 1;
	}
*/
    // When all iterations are done, transfer the result from GPU → CPU
    transfer_3d(u_h, u_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d(f_h, f_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);
    transfer_3d(u_old_h, u_old_d, N + 2, N + 2, N + 2, cudaMemcpyDeviceToHost);

    free_gpu(u_d);
    free_gpu(u_old_d);
    free_gpu(f_d);
}

__global__ void kernel_stop(int N, double ***f, double ***u, double ***u_old, double *d)
{
    double delta = (1.0/(double)N)*(1.0/(double)N);
    int i,j,k;
    double s = 1.0/6.0;  
    double sum = 0.0;
	double value = 0.0;
	 
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i>0 && j>0 && k>0 && j<(N+1) && k<(N+1) && i<(N+1))
	{
    	u[i][j][k]= s*(u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] \
    	+ u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] \
    	+ delta*f[i][j][k]); 
 		
		value = u[i][j][k] - u_old[i][j][k];
		sum += (value * value);   

	}
	//__syncthreads() // kannski!!!
	sum = SumReduce(sum);
	if (threadIdx.x == 0)
	{
		atomicAdd(d, sum);
	}
        
}

__inline__ __device__ double SumReduce(double val)
{
	__shared__ double smem[32];

	if (threadIdx.x < warpSize)
	{
		smem[threadIdx.x] = 0;
	}
	__syncthreads();

	val = WarpReduction(val);

	if (threadIdx.x % warpSize == 0)
	{
		smem[threadIdx.x / warpSize] =val;
	}
	__syncthreads();
	
	if (threadIdx.x < warpSize)
	{
		val = smem[threadIdx.x];
	}		
	return WarpReduction(val);
}

__inline__ __device__ double WarpReduction(double value)
{
	for (int i=16; i > 0; i/2)
	{
		value += __shfl_down_sync(-1, value, i);
	}
	return value;
}


//****************PART 4 - END OF  naive + stop version  *******************

#include "matmult_fun.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C" {
	#include <cblas.h>
	#include <omp.h>

	#define BLOCK_SIZE 11.0;
}

extern "C" {
void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C) {
	double *d_A, *d_B, *d_C;

    /*
    printf("A\n");
    mat_print(m,k,A);
    printf("B\n");
    mat_print(k,n,B);
    */
	
	// set memory on GPU device
	cudaMalloc((void **)&d_C, m * n * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
   
	// Copy data to device
	cudaMemcpy(d_C,C, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B, k * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    
	

	// execute kernel 
	gpu1_kernel<<<1,1>>>(m,n,k,d_A,d_B,d_C);
	cudaDeviceSynchronize();
	
	// transfer results from GPU device
	cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);


	// clean up data on device
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);

	}
}

__global__ void gpu1_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C){

	int i,j,l;
	double sum;

	for (l = 0; l < m; l++)
    {
        for (j = 0; j < n; j++)
        {
            sum = 0.0;
            for (i = 0; i < k; i++)
            {
                sum += d_A[l*k+i] * d_B[i*n+j]; 
            }
            d_C[l*n+j] = sum;
        }
    }
}


extern "C" {
void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C) {
	double *d_A, *d_B, *d_C;
/*
    printf("A\n");
    mat_print(m,k,A);
    printf("B\n");
    mat_print(k,n,B);*/
	
	// set memory on GPU device
	cudaMalloc((void **)&d_C, m * n * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
   
	// Copy data to device
	cudaMemcpy(d_C,C, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B, k * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    
	

	// execute kernel 
	// <NUM_BLOCKS, THREADS PER BLOCK>
	//Number of blocks for each dimensions

	double block_size = 16.0;

	int dim_m = ceil(m/block_size);
	int dim_n = ceil(n/block_size);
	
	dim3 dimGrid(dim_m, dim_n, 1);
	dim3 dimBlock((int)block_size, (int)block_size, 1);
	gpu2_kernel<<<dimGrid, dimBlock>>>(m,n,k,d_A,d_B,d_C);
	checkCudaErrors(cudaDeviceSynchronize());
	
	// transfer results from GPU device
	cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	//mat_print(m,n,C);
	// clean up data on device
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);

	}
}

__global__ void gpu2_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C){

	int j;
	double sum=0.0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//More threads are initialized than needed
	
	if (col < m && row < n)
	{
    		for (j = 0; j < k; j++)
    		{ 
        		sum += d_A[col*k+j] * d_B[j*n+row]; 
    		}
    		d_C[col*n+row] = sum;
	}
}


extern "C" {
void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C) {
	double *d_A, *d_B, *d_C;
/*
    printf("A\n");
    mat_print(m,k,A);
    printf("B\n");
    mat_print(k,n,B);*/
	
	// set memory on GPU device
	cudaMalloc((void **)&d_C, m * n * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
   
	// Copy data to device
	cudaMemcpy(d_C,C, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B, k * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    
	

	// execute kernel 
	// <NUM_BLOCKS, THREADS PER BLOCK>
	//Number of blocks for each dimensions

	double block_size = 4.0;

	int dim_m = ceil(m/block_size);
	int dim_n = ceil(n/block_size);
	
	dim3 dimGrid(dim_m, dim_n, 1);
	dim3 dimBlock((int)block_size, (int)block_size, 1);
	gpu3_kernel<<<dimGrid, dimBlock>>>(m,n,k,d_A,d_B,d_C);
	checkCudaErrors(cudaDeviceSynchronize());
	
	// transfer results from GPU device
	cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	//mat_print(m,n,C);
	// clean up data on device
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);

	}
}

__global__ void gpu3_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C){

	int j;
	double sum1=0.0, sum2=0.0;
	int col = 2*(blockIdx.x * blockDim.x + threadIdx.x);
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//More threads are initialized than needed
	if ((col-1) < m && row < n)
	{
    		for (j = 0; j < k; j++)
    		{ 
        		sum1 += d_A[col*k+j] * d_B[j*n+row];
			sum2 += d_A[(col+1)*k+j] * d_B[j*n+row];
    		}
    		d_C[col*n+row] = sum1;
			d_C[(col+1)*n+row] = sum2;
	
	}
	// If matrix is odd numbered, one calc remaining
	else if (col < m && row < n)
	{
    		for (j = 0; j < k; j++)
    		{ 
        		sum1 += d_A[col*k+j] * d_B[j*n+row];
    		}
    		d_C[col*n+row] = sum1;
	}

}


extern "C" {
void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C) {
	double *d_A, *d_B, *d_C;

    printf("A\n");
    mat_print(m,k,A);
    printf("B\n");
    mat_print(k,n,B);
	
	// set memory on GPU device
	cudaMalloc((void **)&d_C, m * n * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
   
	// Copy data to device
	cudaMemcpy(d_C,C, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B, k * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    
	

	// execute kernel 
	// <NUM_BLOCKS, THREADS PER BLOCK>
	//Number of blocks for each dimensions

	double block_size = 4.0;
	int nr_of_elem = 1;

	int dim_m = ceil(m/block_size);
	int dim_n = ceil(n/block_size);
	
	dim3 dimGrid(dim_m, dim_n, 1);
	dim3 dimBlock((int)block_size, (int)block_size, 1);
	gpu4_kernel<<<dimGrid, dimBlock>>>(m,n,k,d_A,d_B,d_C,nr_of_elem);
	//gpu4_kernel<<<1,1>>>(m,n,k,d_A,d_B,d_C,nr_of_elem);
	checkCudaErrors(cudaDeviceSynchronize());
	
	// transfer results from GPU device
	cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	mat_print(m,n,C);
	// clean up data on device
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);

	}
}

__global__ void gpu4_kernel(int m,int n,int k, double *d_A, double *d_B, double *d_C, int elements){
	
	int j,l;
	double sum=0.0;
	int col = elements*(blockIdx.x * blockDim.x + threadIdx.x);
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int min_el;
	if (elements < (m-col))
	{
		min_el = elements;
	}
	else {min_el = m - col;}

	//More threads are initialized than needed
	if(row<n){
	for (l = 0; l < min_el; l++)
	{
		sum = 0.0;	
    		for (j = 0; j < k; j++)
    		{ 
			sum += d_A[(col+l)*k+j] * d_B[j*n+row];
		}
		
		d_C[(col+l)*n+row] = sum;
	}
}	
}

extern "C" {
void matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    
    //mat_print(m,n,C);
}
}


extern "C" {
void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C) {
   	double *d_A, *d_B, *d_C;
	double alpha=1.0, beta=0.0;
	// set memory on GPU device
    cudaMalloc((void **)&d_C, m * n * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_A, m * k * sizeof(double));

    // Copy data to deviice
    cudaMemcpy(d_C,C, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A,A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n);

	cublasDestroy(handle);
    cudaMemcpy(C, d_C, m*n* sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

    //mat_print(m,n,C);
}
}


extern "C" {
void mat_print(int m, int n, double *A){

	for (int i = 0; i < m; i++)
    	{
        	for (int j=0; j < n; j++)
        	{
            	printf("%.2f     ", A[i*n+j]);
        	}
        	printf("\n");
    	}
	printf("\n\n");

}
}

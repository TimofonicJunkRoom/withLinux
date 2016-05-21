/**
 @file vectorAdd.cu
 @reference cuda samples
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

/**
 * wrapper of void *malloc(size_t size);
 */
inline void *
Malloc(size_t size) {
	void * ret;
	if ((ret = malloc(size)) == NULL) {
		fprintf(stderr, "E: host malloc() failed.\n");
		exit(EXIT_FAILURE);
	}
	return ret;
}

/**
 * __host__ __device__ cudaError_t cudaMalloc ( void** devPtr, size_t size )
 */
__host__ cudaError_t
CudaMalloc (void ** devPtr, size_t size) {
	cudaError_t ret;
	ret = cudaMalloc (devPtr, size);
	if (cudaSuccess != ret) {
        fprintf(stderr, 
			"Failed to allocate device vector A (error code %s)!\n",
			cudaGetErrorString(ret));
        exit(EXIT_FAILURE);
	}
	return ret;
}

/**
 * __host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
 */
__host__ cudaError_t
CudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
	cudaError_t ret;
	ret = cudaMemcpy(dst, src, count, kind);
	if (cudaSuccess != ret) {
        fprintf(stderr,
			"Failed to copy vector A from host to device (error code %s)!\n",
		   	cudaGetErrorString(ret));
        exit(EXIT_FAILURE);
    }
	return ret;
}

/**
 * __host__  __device__ cudaError_t cudaFree ( void* devPtr )
 */
__host__  cudaError_t
CudaFree (void* devPtr) {
	cudaError_t ret;
	ret = cudaFree(devPtr);
    if (cudaSuccess != ret) {
        fprintf(stderr,
			"Failed to free device vector C (error code %s)!\n",
		   	cudaGetErrorString(ret));
        exit(EXIT_FAILURE);
    }
	return ret;
}

/**
 * __host__ cudaError_t cudaDeviceReset ( void )
 */
__host__ cudaError_t cudaDeviceReset ( void )

/**
 * Host main routine
 */
int
main(void)
{
    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("I: Vector addition of %d elements\n", numElements);

    float *h_A = (float *)Malloc(size);
    float *h_B = (float *)Malloc(size);
    float *h_C = (float *)Malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A, B and C
    float *d_A = NULL;
    CudaMalloc((void **)&d_A, size);
    float *d_B = NULL;
    CudaMalloc((void **)&d_B, size);
    float *d_C = NULL;
    CudaMalloc((void **)&d_C, size);

    // Copy the host input vectors A and B in host memory to the
   	// device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    CudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    if (cudaSuccess != cudaGetLastError()) {
        fprintf(stderr, "Failed to launch vectorAdd kernel !\n");
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    CudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // Free device global memory
    CudaFree(d_A);
    CudaFree(d_B);
    CudaFree(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    if (cudaSuccess != cudaDeviceReset()) {
        fprintf(stderr, "Failed to deinitialize the device! error\n");
        exit(EXIT_FAILURE);
    }
    printf("Done\n");
    return 0;
}

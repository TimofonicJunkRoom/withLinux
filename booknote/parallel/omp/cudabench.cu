
#include <cuda_runtime.h>
#include "cudabench.h"

__global__ void
_dcopy_cuda (const double * S, double * D, size_t length)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < length) D[tid] = S[tid];
}

__global__ void
_dscal_cuda (double * x, const double a, size_t n)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) x[tid] = x[tid] * a;
}

__global__ void
_dasum_cuda (const double * a, size_t n, double * s)
{
  //int tid = blockDim.x * blockIdx.x + threadIdx.x;
  *s = .0;
  for (size_t i = 0; i < n; i++)
    *s += (a[i]>0.)?(a[i]):(-a[i]);
}

void
dcopy_cuda (const double * A, double * B, size_t length)
{
  size_t size = sizeof(double) * length;
  // malloc
  double * d_A = NULL, * d_B = NULL;
  cudaMalloc ((void**)&d_A, size);
  cudaMalloc ((void**)&d_B, size);
  // transter H -> D
  cudaMemcpy (d_A, A, size, cudaMemcpyHostToDevice);
  // apply kernel
  int threadsperblock = 256;
  int blockspergrid = (length + threadsperblock - 1)/threadsperblock;
  _dcopy_cuda <<<blockspergrid, threadsperblock>>> (d_A, d_B, length);
  // transter D -> H
  cudaMemcpy (B, d_B, size, cudaMemcpyDeviceToHost);
  // free
  cudaFree (d_A);
  cudaFree (d_B);
}

void
dscal_cuda (double * x, const double a, size_t n)
{
  size_t size = sizeof(double) * n;
  // malloc
  double * d_A = NULL;
  cudaMalloc ((void**)&d_A, size);
  // transter H -> D
  cudaMemcpy (d_A, x, size, cudaMemcpyHostToDevice);
  // apply kernel
  int threadsperblock = 256;
  int blockspergrid = (n + threadsperblock - 1)/threadsperblock;
  _dscal_cuda <<<blockspergrid, threadsperblock>>> (d_A, a, n);
  // transter D -> H
  cudaMemcpy (x, d_A, size, cudaMemcpyDeviceToHost);
  // free
  cudaFree (d_A);
}

double
dasum_cuda (const double * a, size_t n)
{
  size_t size = sizeof(double) * n;
  double s = 0.;
  // malloc
  double * d_A = NULL;
  cudaMalloc ((void**)&d_A, size);
  // transter H -> D
  cudaMemcpy (d_A, a, size, cudaMemcpyHostToDevice);
  // apply kernel
  int threadsperblock = 256;
  int blockspergrid = (n + threadsperblock - 1)/threadsperblock;
  _dasum_cuda <<<blockspergrid, threadsperblock>>> (d_A, n, &s);
  // free
  cudaFree (d_A);
  return s;
}

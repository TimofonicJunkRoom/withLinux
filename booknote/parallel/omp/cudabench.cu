
#include <cuda_runtime.h>
#include "cudabench.h"

__global__ void
_dcopy_cuda (double * S, double * D)
{
  int i = threadIdx.x;
  D[i] = S[i];
}

void
dcopy_cuda (double * A, double * B, size_t length)
{
  size_t size = sizeof(double) * length;
  // malloc
  double * d_A = NULL, * d_B = NULL;
  cudaMalloc ((void**)&d_A, size);
  cudaMalloc ((void**)&d_B, size);
  // transter H -> D
  cudaMemcpy (d_A, A, size, cudaMemcpyHostToDevice);
  // apply kernel
  _dcopy_cuda <<<1, length>>> (A, B);
  // transter D -> H
  cudaMemcpy (B, d_B, size, cudaMemcpyDeviceToHost);
  // free
  cudaFree (d_A);
  cudaFree (d_B);
}

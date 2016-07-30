/*

 plugin for the benchmarking program: benchmark.c

 */
#ifndef CUDABENCH_H_
#define CUDABENCH_H_

#include <cuda_runtime.h>

// kernel functions

__global__ void _dcopy_cuda (double * S, double * D); // double vector copy

// wrapper functions

void dcopy_cuda (double * A, double * B, size_t length);

#endif // CUDABENCH_H_

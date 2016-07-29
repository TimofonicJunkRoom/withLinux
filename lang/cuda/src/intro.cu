/*

 My CUDA intro exercise
 @reference file:///usr/share/doc/nvidia-cuda-doc/html/cuda-c-programming-guide/index.html

 */

#include <iostream>
#include <cuda_runtime.h>

// kernel definition
__global__ void
vectorAdd (float * A, float * B, float * C)
{
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

// kernel definition
__global__ void
matrixAdd (float * A, float * B, float * C, size_t r, size_t c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < r && j < c) {
    C[i*c+j] = A[i*c+j] + B[i*c+j];
  }
}

// kernel wrapper : A + B -> C (vector)
void
WvectorAdd (float * A, float * B, float * C, size_t N)
{
  size_t size = N * sizeof(float);
  // malloc
  std::cout << " - Malloc space for vectors" << std::endl;
  std::cout << "   - " << 3*size << " Bytes required." << std::endl;
  float * d_A = NULL, * d_B = NULL, * d_C = NULL;
  cudaMalloc ((void **)&d_A, size);
  cudaMalloc ((void **)&d_B, size);
  cudaMalloc ((void **)&d_C, size);
  // transfer
  std::cout << " - Copy host memory onto device" << std::endl;
  cudaMemcpy (d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_B, B, size, cudaMemcpyHostToDevice);
  // calculate
  std::cout << " - Apply kernel" << std::endl;
  vectorAdd <<<1, N>>> (d_A, d_B, d_C); // 1 block, N threads
  // transfer
  std::cout << " - Copy device memory back onto host" << std::endl;
  cudaMemcpy (C, d_C, size, cudaMemcpyDeviceToHost);
  // free
  std::cout << " - Free device memory" << std::endl;
  cudaFree (d_A);
  cudaFree (d_B);
  cudaFree (d_C);
}

// kernel wrapper: A + B -> C (matrix)
void
WmatrixAdd (float * A, float * B, float * C, size_t r, size_t c)
{
  size_t size = r * c * sizeof(float);
  // malloc
  float * d_A = NULL, * d_B = NULL, * d_C = NULL;
  cudaMalloc ((void**)&d_A, size);
  cudaMalloc ((void**)&d_B, size);
  cudaMalloc ((void**)&d_C, size);
  // transfer
  cudaMemcpy (d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_B, B, size, cudaMemcpyHostToDevice);
  // calculate
  dim3 threadsperblock (r, c);
  matrixAdd <<<1, threadsperblock>>> (d_A, d_B, d_C, r, c); // 1 block
  // transfer
  cudaMemcpy (C, d_C, size, cudaMemcpyDeviceToHost);
  // free
  cudaFree (d_A);
  cudaFree (d_B);
  cudaFree (d_C);
}

// helper functions
template <typename Tp> void
vectorDump (Tp * v, size_t N)
{
  for (size_t i = 0; i < N; i++) {
    std::cout << v[i] << "\t";
  }
  std::cout << std::endl;
}
template <typename Tp> void
vectorFill (Tp * v, size_t N, Tp stuff)
{
  for (size_t i = 0; i < N; i++) {
    v[i] = stuff;
  }
}
template <typename Tp> void
matrixDump (Tp * m, size_t r, size_t c)
{
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < c; j++) {
      std::cout << m[i*c+j] << "\t";
    }
    std::cout << std::endl;
  }
}
template <typename Tp> void
matrixFill (Tp * m, size_t r, size_t c, Tp stuff)
{
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < c; j++) {
      m[i*c+j] = stuff;
    }
  }
}

int
main (void)
{
  using std::cout;
  using std::endl;

  cout << "GPU is well-suited to address data-parallel computations." << endl;
  cout << "At CUDA's core are three key abstractions:" << endl;
  cout << "  * hierarchy of thread groups" << endl;
  cout << "  * shared memory" << endl;
  cout << "  * barrier synchronization" << endl;
  cout << endl;
  cout << "Partition the problem into coarse sub-problems that can be" << endl;
  cout << "solved independentely in parallel by blocks of threads," << endl;
  cout << "and each sub-problem into finer pieces that can be solved" << endl;
  cout << "cooperatively in parallel by all threads within the block." << endl;

  cout << endl;
  {
    cout << "--[ prepare vectors" << endl;
    float a[10] { .0 };
    float b[10] { .0 };
    float c[10] { .0 };
    vectorFill<float> (a, 10, 1.0);
    vectorFill<float> (b, 10, 1.0);
    vectorDump<float> (a, 10);
    vectorDump<float> (b, 10);
    vectorDump<float> (c, 10);

    cout << "--[ calculate vectoradd" << endl;
    WvectorAdd (a, b, c, 10);
    vectorDump<float> (c, 10);
  }

  cout << endl;
  {
    cout << "--[ prepare matrices" << endl;
    float a[5][5] { 0. };
    float b[5][5] { 0. };
    float c[5][5] { 0. };
    matrixFill<float> ((float *)a, 5, 5, 1.);
    matrixFill<float> ((float *)b, 5, 5, 2.);
    matrixDump<float> ((float *)a, 5, 5);
    matrixDump<float> ((float *)b, 5, 5);
    WmatrixAdd ((float *)a, (float *)b, (float *)c, 5, 5);
    matrixDump<float> ((float *)c, 5, 5);
  }
  
  return 0;
}

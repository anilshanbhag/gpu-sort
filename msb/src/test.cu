// nvcc test.cu sort/*.cu ../external/benchmark/*.cu -O3 -arch=sm_52 -I. -I../external -lcurand

#include "sort/gpu_radix_sort.h"
#include "cub/cub.cuh"
#include <curand.h>

using namespace std;
using namespace cub;

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}


void test_sort_keys(unsigned int num_keys)
{
  uint *d_key_buf;
  uint *d_val_buf;
  uint *d_key_backup;
  uint *d_val_backup;
  uint *d_key_alt_buf;
  uint *d_val_alt_buf;

  CubDebugExit(cudaMalloc((void**)&d_key_buf, sizeof(float) * num_keys));
  CubDebugExit(cudaMalloc((void**)&d_key_backup, sizeof(float) * num_keys));
  CubDebugExit(cudaMalloc((void**)&d_key_alt_buf, sizeof(float) * num_keys));
  CubDebugExit(cudaMalloc((void**)&d_val_buf, sizeof(float) * num_keys));
  CubDebugExit(cudaMalloc((void**)&d_val_backup, sizeof(float) * num_keys));
  CubDebugExit(cudaMalloc((void**)&d_val_alt_buf, sizeof(float) * num_keys));

  curandGenerator_t generator;
  int seed = 0;

  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator,seed);
  curandGenerate(generator, d_key_buf, num_keys);
  curandGenerate(generator, d_val_buf, num_keys);

  cudaMemcpy(d_key_backup, d_key_buf, sizeof(uint) * num_keys, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_val_backup, d_val_buf, sizeof(uint) * num_keys, cudaMemcpyDeviceToDevice);

  SETUP_TIMING();
  int num_trials = 1;
  for (int i=0; i<num_trials; i++) {
    float time_sort_k;
    float time_sort_kv;
    TIME_FUNC((rdxsrt_unstable_sort<uint, cub::NullType, unsigned int>(d_key_buf, NULL, num_keys, d_key_alt_buf, NULL)), time_sort_k);
    cout << "Time Sort K: " << time_sort_k << endl;
    TIME_FUNC((rdxsrt_unstable_sort<uint, uint, unsigned int>(d_key_buf, d_val_buf, num_keys, d_key_alt_buf, d_val_alt_buf, NULL)), time_sort_kv);
    cout << "Time Sort KV: " << time_sort_kv << endl;

    cudaMemcpy(d_key_buf, d_key_backup, sizeof(uint) * num_keys, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_val_buf, d_val_backup, sizeof(uint) * num_keys, cudaMemcpyDeviceToDevice);
  }
}

int main() {
  test_sort_keys(1 << 28);
  return 0;
}

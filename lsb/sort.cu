// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include "cub/test/test_util.h"
#include "gpu_utils.h"

using namespace cub;
using namespace std;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


float sortPairsGPU(float* d_key_buf, float* d_key_alt_buf, uint* d_value_buf, uint  * d_value_alt_buf, int num_items, CachingDeviceAllocator& g_allocator) {
  SETUP_TIMING();

  float time_sort_kv;
  cub::DoubleBuffer<float> d_keys(d_key_buf, d_key_alt_buf);
  cub::DoubleBuffer<uint> d_values(d_value_buf, d_value_alt_buf);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys, d_values, num_items);

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run sorting operation
  TIME_FUNC(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys, d_values, num_items), time_sort_kv);

  CLEANUP(d_temp_storage);
  return time_sort_kv;
}

float sortKeysGPU(float* d_key_buf, float* d_key_alt_buf, int num_items, CachingDeviceAllocator& g_allocator) {
  SETUP_TIMING();

  float time_sort_k;
  cub::DoubleBuffer<float> d_keys(d_key_buf, d_key_alt_buf);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);

  // Allocate temporary storage
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run sorting operation
  TIME_FUNC(cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
      d_keys, num_items), time_sort_k);

  float* res_vec = (float*) malloc(sizeof(float) * 32);
  cudaMemcpy(res_vec, d_keys.Current(), 32 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<32; i++) cout << res_vec[i] << " ";
  cout << endl;

  CLEANUP(d_temp_storage);
  return time_sort_k;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items           = 1<<28;
    int num_trials          = 3;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("t", num_trials);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items>] "
            "[--t=<num trials>] "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Allocate problem device arrays
    float *d_key_buf;
    float *d_key_alt_buf;
    float *d_key_backup;
    uint   *d_value_buf;
    uint   *d_value_alt_buf;
    uint   *d_value_backup;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_key_buf, sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_key_alt_buf, sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_key_backup, sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_value_buf, sizeof(uint) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_value_alt_buf, sizeof(uint) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_value_backup, sizeof(uint) * num_items));

    curandGenerator_t generator;
    int seed = 0;

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    curandGenerateUniform(generator, d_key_buf, num_items);
    curandGenerate(generator, d_value_buf, num_items);

    cudaMemcpy(d_key_backup, d_key_buf, sizeof(float) * num_items, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_value_backup, d_value_buf, sizeof(uint) * num_items, cudaMemcpyDeviceToDevice);

    float time_sort_kv_gpu, time_sort_k_gpu;

    for (int t = 0; t < num_trials; t++) {
        time_sort_kv_gpu = sortPairsGPU(d_key_buf, d_key_alt_buf, d_value_buf, d_value_alt_buf, num_items, g_allocator);

        cudaMemcpy(d_key_buf, d_key_backup, sizeof(float) * num_items, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_value_buf, d_value_backup, sizeof(uint) * num_items, cudaMemcpyDeviceToDevice);

        time_sort_k_gpu = sortKeysGPU(d_key_buf, d_key_alt_buf, num_items, g_allocator);

        cudaMemcpy(d_key_buf, d_key_backup, sizeof(float) * num_items, cudaMemcpyDeviceToDevice);

        cout<< "{"
            << "\"time_sort_kv_gpu\":" << time_sort_kv_gpu
            << ",\"time_sort_k_gpu\":" << time_sort_k_gpu
            << "}" << endl;
    }

    // Cleanup
    if (d_key_buf) CubDebugExit(g_allocator.DeviceFree(d_key_buf));
    if (d_key_alt_buf) CubDebugExit(g_allocator.DeviceFree(d_key_alt_buf));
    if (d_key_backup) CubDebugExit(g_allocator.DeviceFree(d_key_backup));

    if (d_value_buf) CubDebugExit(g_allocator.DeviceFree(d_value_buf));
    if (d_value_alt_buf) CubDebugExit(g_allocator.DeviceFree(d_value_alt_buf));
    if (d_value_backup) CubDebugExit(g_allocator.DeviceFree(d_value_backup));

    return 0;
}

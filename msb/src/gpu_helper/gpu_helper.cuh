#ifndef GPU_HELPER_CUH_
#define GPU_HELPER_CUH_

#include "gpu_helper/gpu_warmup.cuh"
#include <iostream>

inline size_t print_gpu_info()
{
	size_t total_mem, free_mem;
	printf("-- DEVICE INFO --\n");
	cudaMemGetInfo(&free_mem, &total_mem);
	printf(" - Free Global Memory:  %10zu\n", free_mem);
	printf(" - Total Global Memory: %10zu\n", total_mem);
	printf("-- DEVICE INFO --\n\n");
	return free_mem;
}

template<typename T>
inline void print_gpu_array_values(T *dev_array, size_t num_items)
{
	T *items = (T*)malloc(num_items * sizeof(T));
	cudaMemcpy(items, dev_array, num_items*sizeof(T), cudaMemcpyDeviceToHost);
	for(size_t i=0; i<num_items; i++){
		std::cout << i << ": " << items[i] << "\n";
	}
	free(items);
}


#endif /* GPU_HELPER_CUH_ */

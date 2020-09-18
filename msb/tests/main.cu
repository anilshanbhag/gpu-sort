#include <unistd.h>
#include "gtest/gtest.h"
#include "gpu_helper/gpu_helper.cuh"
#include "benchmark/benchmark.h"
#include "cli_args.h"

unsigned int sort_repeat_count;
unsigned int sort_keys_prob_size_in_mb;
unsigned int sort_keys_default_prob_size;
unsigned int sort_pairs_default_prob_size;

int main(int argc, char **argv)
{
	// Make print buffer large enough
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*4);

	/*****************************************
	 *** GPU DEVICE OVERVIEW AND SELECTION ***
	 *****************************************/
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	printf("\n --- DEVICES ---\n");
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf(" -> device %d (%s) has %d SMs (shared memory: %zu B, registers: %6d) @%d MHz, memory-bus: %d bit @%d MHz (%d GB/s). \n", device, deviceProp.name, deviceProp.multiProcessorCount, deviceProp.sharedMemPerMultiprocessor, deviceProp.regsPerMultiprocessor, deviceProp.clockRate/1000, deviceProp.memoryBusWidth, deviceProp.memoryClockRate/1000, deviceProp.memoryBusWidth/8*(deviceProp.memoryClockRate)*2/1000000);
	}
	printf(" --- DEVICES ---\n");
	int selected_device = 0;
	cudaSetDevice(selected_device);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, selected_device);
	printf(" -> Selected device %d (%s) has %d SMs (shared memory: %zu B, registers: %6d) @%d MHz, memory-bus: %d bit @%d MHz (%d GB/s). \n", selected_device, deviceProp.name, deviceProp.multiProcessorCount, deviceProp.sharedMemPerMultiprocessor, deviceProp.regsPerMultiprocessor, deviceProp.clockRate/1000, deviceProp.memoryBusWidth, deviceProp.memoryClockRate/1000, deviceProp.memoryBusWidth/8*(deviceProp.memoryClockRate)*2/1000000);
	print_gpu_info();

	/******************************
	 *** CLI ARGUMENTS  PARSING ***
	 ******************************/
	opterr = 0;
	sort_keys_prob_size_in_mb = 0;
	sort_keys_default_prob_size = 200000;
	sort_pairs_default_prob_size = 100000;
	sort_repeat_count = 3;
	int c;
	while ((c = getopt(argc, argv, "r:k:p:s:")) != -1) {
		switch (c) {
			case 'r': sort_repeat_count = atoi(optarg); 			break;
			case 'k': sort_keys_default_prob_size = atoi(optarg); 	break;
			case 'p': sort_pairs_default_prob_size = atoi(optarg); 	break;
			case 's': sort_keys_prob_size_in_mb = atoi(optarg); 	break;
			default: break;
		}
	}

	/******************************
	 *** DEFAULT TEST SELECTION ***
	 ******************************/
	::testing::InitGoogleTest(&argc, argv);
	int test_result = RUN_ALL_TESTS();

	/******************************
	 ***   BENCHMARK OVERVIEW   ***
	 ******************************/
	if(::testing::UnitTest::GetInstance()->test_to_run_count()>0){
		printf("\n --- PROFILE OVERVIEW ---\n");
		BM_PRINT_ALL_PROFILES('\t');
		printf(" --- PROFILE OVERVIEW ---\n");
	}

	return test_result;
}

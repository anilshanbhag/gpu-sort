#include <stdio.h>
#include "sort/cuda_radix_sort_config.h"
#include "sort/cuda_radix_sort.h"


__global__ void do_compute_locrec_segmented_assignment_offsets(unsigned int *locrec_per_kpb_atomic_counter_array, unsigned int *locrec_per_kpb_assnmt_offsets)
{
	unsigned int locrec_bin_assinmt_offsets[RDXSRT_MAX_NUM_SORT_CONFIGS];
	if(threadIdx.x == 0){
		#pragma unroll
		for(int i= 0; i < RDXSRT_MAX_NUM_SORT_CONFIGS; i++){
			locrec_bin_assinmt_offsets[i] = locrec_per_kpb_atomic_counter_array[i];
//			printf(" Local Sort Config #%2d: %5d sub-buckets\n", i, locrec_bin_assinmt_offsets[i]);
		}

		unsigned int tmp_pfx_sum = 0;
		unsigned int tmp = 0;
		#pragma unroll
		for(int i= 0; i < RDXSRT_MAX_NUM_SORT_CONFIGS; i++){
			tmp = locrec_bin_assinmt_offsets[i];
			locrec_bin_assinmt_offsets[i] = tmp_pfx_sum;
			tmp_pfx_sum += tmp;
		}

		#pragma unroll
		for(int i= 0; i < RDXSRT_MAX_NUM_SORT_CONFIGS; i++)
			locrec_per_kpb_assnmt_offsets[i] = locrec_bin_assinmt_offsets[i];
	}
}

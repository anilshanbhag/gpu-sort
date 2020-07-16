#ifndef CUDA_RADIX_SORT_CONFIG_H_
#define CUDA_RADIX_SORT_CONFIG_H_

#define RDXSRT_WARP_THREADS 32

#define RDXSRT_MAX_NUM_SORT_CONFIGS 16

// 0 for no merging. N for merge if num-keys in bucket is below threshold N
#define RDXSRT_CFG_MERGE_LOCREC_THRESH 3000

#endif /* CUDA_RADIX_SORT_CONFIG_H_ */

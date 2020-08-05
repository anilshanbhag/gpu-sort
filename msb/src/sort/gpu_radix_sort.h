#ifndef GPU_RADIX_SORT_H_
#define GPU_RADIX_SORT_H_

#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include "cub/cub.cuh"
#include "gpu_helper/gpu_helper.cuh"
#include "sort/gpu_sort_config.h"
#include "sort/cuda_radix_sort_common.h"
#include "sort/cuda_radix_sort_config.h"
#include "sort/cuda_radix_sort.h"


/*** PRIVATE ONES ****/
struct extsrt_assignment_info_t {
	unsigned int num_ext_bins;
	unsigned int num_extsrt_block_count;
	unsigned int num_extsrt_remainder_block_count;
	unsigned int extsrt_full_block_offset;
};
void rdxsrt_prepare_histograms(unsigned int *dev_histo, unsigned int num_ext_in_bins, cudaStream_t cstream);
struct extsrt_assignment_info_t generate_next_pass_block_assignments(const unsigned int *dev_atomic_extsrt_bucket_counter, const SubBucketInfo *dev_extsrt_in_bin_info, SubBucketInfo *extsrt_in_bin_info,  struct rdxsrt_extsrt_block_to_bin_t *extsrt_blck_to_in_bin_assignments, unsigned int max_ext_srt_kpb, cudaStream_t cstream, cudaEvent_t offst_comp_event);

// void rdxsrt_prepare_histograms(unsigned int *dev_histo, unsigned int num_ext_in_bins, cudaStream_t cstream)
// {
// 	// For every IN-bin that needs to be sorted externally, we need one histogram with each 256 sub-bins
// 	if(num_ext_in_bins>0)
// 		cudaMemsetAsync(dev_histo, 0, num_ext_in_bins * 256 * sizeof(*dev_histo), cstream);
// }

// /** *
//  * Takes as input the number of extsrt OUT-bins of the last pass (dev_atomic_ext_srt_counter), along with info on on their unique index, offset and number of elements (dev_extsrt_in_bin_info),
//  * copies the information to extsrt_in_bin_info and computes the block assignments accordingly, writing it to extsrt_blck_to_in_bin_assignments.
//  *
//  * @param dev_atomic_ext_srt_bin_counter		The number of EXTSRT-OUT-bins from the last pass. Equals the number of EXTSRT-IN-bins of this pass.
//  * @param dev_extsrt_in_bin_info				Information about the EXTSRT-OUT-bins from the last pass, such as a unique index (bin_idx), the bin's offset within the *keys array (offset) and the number of keys in that bin (block_num_elements).
//  * @param extsrt_in_bin_info					Pointer to host side memory to which dev_extsrt_in_bin_info should be copied to.
//  * @param extsrt_blck_to_in_bin_assignments		Pointer to memory to which the block assignments that are computed according to dev_extsrt_in_bin_info should be written to.
//  * @param max_ext_srt_kpb						The maximum number of keys each extsrt-block will cover
//  * @param cstream								The cuda stream to use for copying the device side information of the EXTSRT-bins to the Host
//  * @param offst_comp_event						The event to synchronize with, such that it is guaranteed that the information in dev_atomic_ext_srt_bin_counter and dev_extsrt_in_bin_info is final.
//  * @return	Returns the number of EXTSRT-bins and the total number of EXTSRT-blocks required to externally radix-sort the EXTSRT-bins.
//  */
// struct extsrt_assignment_info_t generate_next_pass_block_assignments(const unsigned int *dev_atomic_extsrt_bucket_counter, const SubBucketInfo *dev_extsrt_in_bin_info, SubBucketInfo *extsrt_in_bin_info,  struct rdxsrt_extsrt_block_to_bin_t *extsrt_blck_to_in_bin_assignments, unsigned int max_ext_srt_kpb, cudaStream_t cstream, cudaEvent_t offst_comp_event){
// 	// Make sure GPU has finished computing the sub-buckets of the last pass
// 	cudaEventSynchronize(offst_comp_event);

// 	// Get number of sub-buckets that require another counting sort
// 	unsigned int num_next_pass_ext_buckets;
// 	cudaMemcpyAsync(&num_next_pass_ext_buckets, dev_atomic_extsrt_bucket_counter, sizeof(num_next_pass_ext_buckets), cudaMemcpyDeviceToHost, cstream);
// //	printf("Num next pass buckets: %u\n", num_next_pass_ext_buckets);
// 	cudaStreamSynchronize(cstream); // TODO think of giving it it's own stream, if that makes sense: look at the dependency - we want to interleave CPU preparation and copy, in best case with local sort

// 	// Get info (.offset: sub-bucket offset, .block_num_elements: total number of sub-bucket's keys) on the sub-buckets that require another counting sort
// 	if(num_next_pass_ext_buckets==0){
// 		struct extsrt_assignment_info_t tmp;
// 		tmp.num_ext_bins = 0;
// 		tmp.num_extsrt_block_count = 0;
// 		tmp.num_extsrt_remainder_block_count = 0;
// 		tmp.extsrt_full_block_offset = 0;
// 		return tmp;
// 	}
// 	cudaMemcpyAsync(extsrt_in_bin_info, dev_extsrt_in_bin_info, num_next_pass_ext_buckets * sizeof(*extsrt_in_bin_info), cudaMemcpyDeviceToHost, cstream);
// 	cudaStreamSynchronize(cstream);

// 	// Counting the required number of full blocks for the counting sorts of the coming pass
// 	unsigned int num_next_pass_ext_block_count = 0;
// 	// Counting the required number of remainder blocks (not-full) for the counting sorts of the coming pass
// 	unsigned int num_extsrt_remainder_block_count = 0;

// 	// The first num_next_pass_ext_buckets assignments are reserved for the up to num_next_pass_ext_buckets remainder blocks, such that in memory we have:
// 	// <---REMAINDER BLOCK ASSIGNMENTS---> <------------ num_next_pass_ext_block_count FULL BLOCK ASSIGNMENTS ------------------->
// 	// [0,1,..,num_next_pass_ext_buckets-1,num_next_pass_ext_buckets,...,num_next_pass_ext_block_count+num_next_pass_ext_buckets-1]
// 	unsigned int full_block_offset = num_next_pass_ext_buckets;

// 	// Prepare one assignment for each single block involved in processing the counting sort for one of the buckets
// //	START_CPU_TIMER(1, bm_cpu_genass);
// //	printf("START\n");
// 	for(unsigned int i=0; i < num_next_pass_ext_buckets; i++){
// 		SubBucketInfo tmp;
// 		tmp = extsrt_in_bin_info[i];
// 		unsigned int bin_offset = tmp.offset;
// //		printf("next pass bucket %5d (id, offset, size): %u, %u, %u\n", i, tmp.bin_idx, tmp.offset, tmp.block_num_elements);
// //		printf("%u, %u, %u\n", tmp.bin_idx, tmp.offset, tmp.block_num_elements);
// 		for(unsigned int j=0; tmp.block_num_elements > 0; j++){
// 			// Another full block
// 			if(tmp.block_num_elements >= max_ext_srt_kpb){
// 				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].bin_idx = tmp.bin_idx;
// 				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].offset = tmp.offset;
// 				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].bin_start_offset = bin_offset;
// 				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].block_num_elements = max_ext_srt_kpb;

// 				tmp.offset += extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].block_num_elements;
// 				tmp.block_num_elements -= extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].block_num_elements;
// 				num_next_pass_ext_block_count++;
// 			}
// 			// The remainder block of that bucket
// 			else{
// 				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].bin_idx = tmp.bin_idx;
// 				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].offset = tmp.offset;
// 				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].bin_start_offset = bin_offset;
// 				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].block_num_elements = tmp.block_num_elements > max_ext_srt_kpb ? max_ext_srt_kpb : tmp.block_num_elements;

// 				tmp.offset += extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].block_num_elements;
// 				tmp.block_num_elements -= extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].block_num_elements;
// 				num_extsrt_remainder_block_count++;
// 			}
// 		}
// 	}
// //	printf("END\n");
// //	STOP_CPU_TIMER(bm_cpu_genass, "CPU prepare block assignments: ");

// 	struct extsrt_assignment_info_t tmp;
// 	tmp.num_ext_bins = num_next_pass_ext_buckets;
// 	tmp.num_extsrt_block_count = num_next_pass_ext_block_count;
// 	tmp.num_extsrt_remainder_block_count = num_extsrt_remainder_block_count;
// 	tmp.extsrt_full_block_offset = full_block_offset;
// 	return tmp;
// }

template <
	typename KeyT,						// Key type
	typename ValueT = cub::NullType		// Value type (for keys-only, pass cub::NullType)
>
struct RDXSRT_SortedSequence {
	KeyT *sorted_keys;
	ValueT *sorted_values;
};

struct NextPassBucketCounter {
	unsigned int dev_nl_bucket_counter; //dev_atomic_ext_srt_counter; -> dm->bucket_counters[pass].dev_nl_bucket_counter
	unsigned int dev_local_bucket_counter[RDXSRT_MAX_NUM_SORT_CONFIGS]; //dev_atomic_locrec_srt_counter[pass][config];
	unsigned int dev_local_cfg_offsets[RDXSRT_MAX_NUM_SORT_CONFIGS]; //dev_atomic_locrec_assnmt_offsets;
};

template<
	typename KeyT,
	typename ValueT,
	typename IndexT = unsigned int,
	unsigned int TPB = DefaultRadixSortConfig<KeyT, ValueT>::TPB,	// The number of threads per block to be used for the partitioning kernel
	unsigned int KPT = DefaultRadixSortConfig<KeyT, ValueT>::KPT,	// The number of keys per thread to be used for the partitioning kernel
	unsigned int DIGIT_BITS = 8
>
struct RDXSRT_GPUDataManager {

	enum { RADIX = (0x01<<DIGIT_BITS) };
	enum { KPB = TPB*KPT };
	typedef unsigned int PerBlockIndexT;

	const unsigned int max_num_passes;

	/** UPPER BOUNDS ON BUCKETS ET AL ***/
	const IndexT max_nl_buckets;		// Maximum number of buckets that cannot be sorted locally
	const IndexT max_nl_blocks;			// Maximum number of blocks, involved in sorting buckets that cannot be sorted locally
	const IndexT max_local_buckets;

	/*** UPPER BOUNDS ON MEMORY REQUIREMENTS ***/
	const size_t max_histogram_mem;				// Maximum memory required to store the histograms of the buckets that are not sorted locally
	const size_t max_per_block_histogram_mem;	//
	const size_t max_block_assignments_mem;		//
	const size_t max_dev_subbucket_info_mem;	//
	const size_t max_dev_local_bucket_info_mem;	//
	const size_t max_counter_mem;				//

	/*** DEVICE MEMORY POINTERS ***/
	IndexT *dev_histogram;				// Memory to store the histograms non-local buckets (one counter for each of the buckets' sub-buckets)
	IndexT *dev_subbucket_offsets;		// Memory to store the offsets of all non-local buckets' sub-buckets
	PerBlockIndexT *dev_per_block_histogram;	// Memory to store the offsets of all non-local buckets' sub-buckets

	rdxsrt_extsrt_block_to_bin_t *dev_nl_block_assignments;	//

	SubBucketInfo *dev_nl_bucket_info;		//
	SubBucketInfo *dev_nl_subbucket_info;	//
	rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> *dev_local_bucket_info; //

	NextPassBucketCounter *bucket_counters; //

	/*** HOST MEMORY POINTERS ***/
	rdxsrt_extsrt_block_to_bin_t *nl_block_assignments;	// dev_nl_block_assignments's counterpart on the host, only used temporarily for preparation
	SubBucketInfo *nl_bucket_info_buffer;	//


	RDXSRT_GPUDataManager(IndexT key_count, LocalSortConfigSet<KeyT, ValueT> *local_sort_configurations, unsigned int max_num_passes = (DIGIT_BITS-1 + sizeof(KeyT)*8)/DIGIT_BITS) :
		max_nl_buckets( std::max(1U, key_count/local_sort_configurations->max_kpb()) ),		// At least one bucket required, i.e., the bucket for the input (pass 0)
		max_nl_blocks( key_count/KPB + max_nl_buckets ),									//
		max_local_buckets( (RDXSRT_CFG_MERGE_LOCREC_THRESH>0) ? (min(2*key_count / RDXSRT_CFG_MERGE_LOCREC_THRESH + max_nl_buckets, max_nl_buckets*RADIX)) : (max_nl_buckets*RADIX) ),
		max_histogram_mem(max_nl_buckets * RADIX * sizeof(IndexT)),							// One histogram of RADIX counters for each possible non-local bucket
		max_per_block_histogram_mem(max_nl_blocks * RADIX * sizeof(IndexT)),				// One histogram of RADIX counters for every block of a non-local bucket
		max_block_assignments_mem(max_nl_blocks * sizeof(rdxsrt_extsrt_block_to_bin_t)),	// One data-structure with information on the block's bucket ID, the block's input range and the buckets offset
		max_dev_subbucket_info_mem(max_nl_buckets * sizeof(SubBucketInfo)),					// One data-structure for every possible non-local bucket, to hold information on the bucket's id, offset and size
		max_dev_local_bucket_info_mem(max_local_buckets * sizeof(rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH>)),	//
		max_num_passes(max_num_passes),
		max_counter_mem(max_num_passes * sizeof(NextPassBucketCounter))	// Atomic counters for the number of local and non-local sub-buckets for the (NUM_PASSES-1) passes
	{
//		printf("\n--- DATA FOR GPU-RADIX SORT MANAGAMENT INFO ---\n");
//		printf("%-.40s: %10u %s\n", "Max. non-local buckets", max_nl_buckets, "buckets");
//		printf("%-40s: %10u %s\n", "Max. local buckets", max_local_buckets, "small buckets");
//		printf("%-40s: %10u %s\n", "Max. non-local blocks", max_nl_blocks, "small buckets");
//		printf("%-40s: %10zu %s\n", "Max. bucket counter memory", max_counter_mem, "Byte");
//		printf("%-40s: %10zu %s\n", "Max. local bucket info memory", max_dev_subbucket_info_mem, "Byte");

		// Per non-local bucket histogram memory
		cudaMalloc(&dev_histogram, max_histogram_mem);

		// Memory for the offsets of each of the RADIX sub-buckets of each non-local bucket
		cudaMalloc(&dev_subbucket_offsets, max_histogram_mem);

		// Memory for one histogram computed by each block involved in processing a non-local bucket's histogram
		cudaMalloc(&dev_per_block_histogram, max_per_block_histogram_mem);

		// Memory used to assign each (thread-)block to a key-block (a sequence of keys within a bucket)
		cudaMalloc(&dev_nl_block_assignments, max_block_assignments_mem);

		// Allocate memory for double buffer, which has information on current pass's buckets (with constant look-up complexity)
		// and provides memory to store information on the bucket's sub-buckets, like offset, bucketId and number of keys
		cudaMalloc(&dev_nl_bucket_info, max_dev_subbucket_info_mem);
		cudaMalloc(&dev_nl_subbucket_info, max_dev_subbucket_info_mem);

		// Memory for the information (offset and size) of small buckets that can be sorted using a local sort
		cudaMalloc(&dev_local_bucket_info, max_dev_local_bucket_info_mem);

		// Memory used to
		cudaMalloc(&bucket_counters, max_counter_mem);

//		size_t total_device_memory = max_histogram_mem + max_histogram_mem + max_per_block_histogram_mem + max_block_assignments_mem + max_dev_subbucket_info_mem + max_dev_subbucket_info_mem + max_local_buckets + max_counter_mem;
//		printf(" Total Device memory allocation: %zu\n", total_device_memory);
//		size_t total_host_memory = max_block_assignments_mem + max_dev_subbucket_info_mem;
//		printf(" Total host memory allocation: %zu\n", total_host_memory);

		/*** HOST SIDE MEMORY ALLOCATIONS ***/
//		cudaMallocHost(&nl_block_assignments, max_block_assignments_mem);
//		cudaMallocHost(&nl_bucket_info_buffer, max_dev_subbucket_info_mem);
		nl_block_assignments = (rdxsrt_extsrt_block_to_bin_t *)malloc(max_block_assignments_mem);
		nl_bucket_info_buffer = (SubBucketInfo *)malloc(max_dev_subbucket_info_mem);
	}

	~RDXSRT_GPUDataManager()
	{
		cudaFree(dev_histogram);
		cudaFree(dev_subbucket_offsets);
		cudaFree(dev_per_block_histogram);
		cudaFree(dev_nl_block_assignments);
		cudaFree(dev_nl_subbucket_info);
		cudaFree(dev_nl_bucket_info);
		cudaFree(dev_local_bucket_info);
		cudaFree(bucket_counters);

		free(nl_block_assignments);
		free(nl_bucket_info_buffer);
//		cudaFreeHost(nl_block_assignments);
//		cudaFreeHost(nl_bucket_info_buffer);
	}


	void init_memory()
	{

	}

};

/*************************
 ***   GPU INTERFACE   ***
 *************************/
template<
	bool PERFORM_SWAP
>
struct SWAPPER{
	template<typename KeyT, typename ValueT>
	static void key_lsb_in_value_pointer_swap(KeyT &key_ptr, ValueT &value_ptr){ }
};

template<>
struct SWAPPER<true>{
	template<typename T>
	static void key_lsb_in_value_pointer_swap(T &key_ptr, T &value_ptr){
		std::swap<T>(key_ptr, value_ptr);
	}
};

template <
	typename KeyT,						// Key type
	typename ValueT = cub::NullType,	// Value type (for keys-only, pass cub::NullType)
	typename IndexT = unsigned int,		// The index type to be used (i.e. unsigned int for up to 2^32-1 keys)
	unsigned int TPB = DefaultRadixSortConfig<KeyT, ValueT>::TPB,	// The number of threads per block to be used for the partitioning kernel
	unsigned int KPT = DefaultRadixSortConfig<KeyT, ValueT>::KPT,	// The number of keys per thread to be used for the partitioning kernel
	unsigned int DIGIT_BITS = 8,		// The number of bits to sort with each partitioning pass
	unsigned int TINY_BUCKET_MERGE_THRESHOLD = 3000,
	unsigned int NUM_LSB_IN_VALUE = 0
>
RDXSRT_SortedSequence<KeyT, ValueT> rdxsrt_unstable_sort(KeyT *dev_keys, ValueT *dev_values, IndexT key_count, KeyT *dev_sorted_keys_out, ValueT *dev_sorted_values_out, LocalSortConfigSet<KeyT, ValueT> *local_sort_configurations = NULL, RDXSRT_GPUDataManager<KeyT, ValueT, IndexT, TPB, KPT, DIGIT_BITS> *pre_allocated_dm = NULL, cudaStream_t cstrm_extsrt = NULL)
{
#define DBG_LEV 0

	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;

	const unsigned int RADIX = (0x01<<DIGIT_BITS);
	const unsigned int NUM_KEY_PASSES = (DIGIT_BITS-1 + sizeof(KeyT)*8)/DIGIT_BITS;
	const unsigned int NUM_LSB_IN_VAL_PASSES = (DIGIT_BITS-1 + NUM_LSB_IN_VALUE*8)/DIGIT_BITS;
	const unsigned int NUM_PASSES = NUM_KEY_PASSES + NUM_LSB_IN_VAL_PASSES;
	const int KPB = TPB * KPT;	// Keys per block (for ext-sort)

	//BM_START_CPU_PROFILE(sort configs, sort_configs);
	/**** PREPARE LOCAL SORT CONFIGURATIONS ****/
	if(!local_sort_configurations){
//		printf("Getting default local sort configs\n");
		local_sort_configurations = LocalSortConfigSet<KeyT, ValueT>::GetDefaultConfigSet();
	}

	int num_local_sort_cfgs = local_sort_configurations->num_configs();
	int local_sort_max_kpb = local_sort_configurations->max_kpb();
	int  *dev_locrec_distinctions  = local_sort_configurations->GetDeviceLocalSortConfigThresholds();
	assert(RDXSRT_CFG_MERGE_LOCREC_THRESH<=local_sort_max_kpb);
	/**** PREPARE LOCAL SORT CONFIGURATIONS ****/

	/**** PREPARE MEMORY ALLOCATION & INITIALISATION ****/
	RDXSRT_GPUDataManager<KeyT, ValueT, IndexT, TPB, KPT, DIGIT_BITS> *dm;
	if(!pre_allocated_dm)
		dm = new RDXSRT_GPUDataManager<KeyT, ValueT, IndexT, TPB, KPT, DIGIT_BITS>(key_count, local_sort_configurations, 1+((sizeof(KeyT)+NUM_LSB_IN_VALUE)*8-1)/DIGIT_BITS);
	else
		dm = pre_allocated_dm;
	/**** PREPARE MEMORY ALLOCATION & INITIALISATION ****/

	// Initialize for Pass-0 sorting (MSB / MSD)
	unsigned int num_blocks = key_count / KPB;							// Pass-0 rough processing blocks (floor on purpose)
	IndexT processed_elements = num_blocks * KPB;						// Pass-0 number of rough processed elements
	IndexT remaining_elements = key_count - processed_elements;			// Do the remaining elements with a check in the inner loop
	unsigned int remainder_blocks = (KPB-1+remaining_elements) / KPB;	// Number of blocks required for remaining elements (typically 0 or 1)

//	print_gpu_info();


	/*** PREPARE CUDA STREAMS ***/
	cudaStream_t cstrm_prep_nxtp, cstrm_lsort;
	cudaEvent_t cstrm_nxtp_assignment_compl[NUM_PASSES];
	cudaEvent_t cstrm_lastp_nlsort_compl[NUM_PASSES];
	cudaEvent_t cstrm_local_sort_compl[NUM_PASSES];
	for(int i=0; i < NUM_PASSES; i++){
		cudaEventCreate(&cstrm_nxtp_assignment_compl[i]);
		cudaEventCreate(&cstrm_lastp_nlsort_compl[i]);
		cudaEventCreate(&cstrm_local_sort_compl[i]);
	}
	bool own_extstrt_stream = false;
	if(cstrm_extsrt == NULL){
		cudaStreamCreate(&cstrm_extsrt);
		own_extstrt_stream = true;
	}
	cudaStreamCreate(&cstrm_prep_nxtp);
	cudaStreamCreate(&cstrm_lsort);
	/*** PREPARE CUDA STREAMS ***/

	/*** START PROCESSING BENCHMARK ***/
	//BM_START_CUDA_EVENT(sort processing, sort_proc, cstrm_extsrt);
	/*** START PROCESSING BENCHMARK ***/

	/*** INITIALIZE COUNTERS ***/
	cudaMemsetAsync(dm->bucket_counters, 0, dm->max_counter_mem, cstrm_extsrt);

	//BM_DECLARE_METRIC_ARRAY(histo, histo, 0, 8);
	//BM_DECLARE_METRIC_ARRAY(pfx_sum, pfx_sum, 0, 8);
	//BM_DECLARE_METRIC_ARRAY(scatter, scatter, 0, 8);
	//BM_DECLARE_METRIC_ARRAY(local_sort, local_sort, 0, 8);

	/***************************
	 ***************************
	 *********  PASS 0  ********
	 ***************************
	 ***************************/

	/********* HISTOGRAM PASS 0 *********/
	//BM_START_CUDA_ARRAY_EVENT_LEV(1, histo, histo, cstrm_extsrt, 0, 8);
	rdxsrt_prepare_histograms(dm->dev_histogram, 1, cstrm_extsrt);
	rdxsrt_histogram<KeyT, IndexT, DIGIT_BITS, KPT, TPB, 9, true><<<num_blocks, TPB, 0, cstrm_extsrt>>>(dev_keys, NULL, 0, dm->dev_histogram, dm->dev_per_block_histogram);
	if(remaining_elements > 0){
		rdxsrt_histogram_with_guards<KeyT, IndexT, DIGIT_BITS, KPT, TPB, 9, true><<<remainder_blocks, TPB, 0, cstrm_extsrt>>>(dev_keys, NULL, 0, dm->dev_histogram, dm->dev_per_block_histogram, key_count, num_blocks);
	}
	//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, histo, histo, cstrm_extsrt, 0, 8);
//	rdxsrt_histogram_with_guards<KeyT, IndexT, DIGIT_BITS, KPT, TPB, 9, true><<<num_blocks+remainder_blocks, TPB, 0, cstrm_extsrt>>>(dev_keys, NULL, 0, dm->dev_histogram, dm->dev_per_block_histogram, key_count, 0);
	/********* HISTOGRAM PASS 0 *********/

	/********* COMPUTE OFFSETS OF OUT-BINS PASS 0 AND ASSIGNMENTS OF PASS 1 *********/
	//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, pfx_sum, pfx_sum, cstrm_extsrt, 0, 8);
	// Prepare assignment for the only IN-bin in the first pass
	dm->nl_bucket_info_buffer[0] = (SubBucketInfo){0};
	cudaMemcpyAsync(dm->dev_nl_bucket_info, dm->nl_bucket_info_buffer, 1 * sizeof(SubBucketInfo), cudaMemcpyHostToDevice, cstrm_extsrt);
	// Compute current pass out-bin offsets and next pass' block assignments
	#if RDXSRT_CFG_MERGE_LOCREC_THRESH
	do_fast_merged_compute_subbin_offsets_and_segmented_next_pass_assignments<IndexT, RADIX, 8, 128, RDXSRT_MAX_NUM_SORT_CONFIGS, RDXSRT_CFG_MERGE_LOCREC_THRESH, true, true><<<1, 128, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[0].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[0].dev_local_bucket_counter[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);

#if false || DEBUG_INFI
	unsigned int locsort_cfg_counts[num_local_sort_cfgs];
	rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> *locsrt_bucket = (rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> *)malloc(dm->max_dev_local_bucket_info_mem);
	cudaMemcpy(locsort_cfg_counts, &(dm->bucket_counters[0].dev_local_bucket_counter[0]), num_local_sort_cfgs*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for(unsigned int i=0;i<num_local_sort_cfgs;i++){
		printf("Num. sub-buckets for local sort config #%2d, %7u \n", i, locsort_cfg_counts[i]);

		if(locsort_cfg_counts[i]>0)
			cudaMemcpy(locsrt_bucket, dm->dev_local_bucket_info, locsort_cfg_counts[i]*sizeof(*dm->dev_local_bucket_info), cudaMemcpyDeviceToHost);
		unsigned int num_merged = 0;
		unsigned int min_merged = 0;
		unsigned int max_merged = 0;
		unsigned int sum_merged = 0;
		unsigned int min = 0;
		unsigned int max = 0;
		unsigned int sum = 0;
		for(unsigned int j=0;j<locsort_cfg_counts[i];j++){
			sum += locsrt_bucket[j].num_elements;
			num_merged += locsrt_bucket[j].is_merged;
			if(locsrt_bucket[j].is_merged){
				min_merged = min_merged>locsrt_bucket[j].num_elements?locsrt_bucket[j].num_elements:min_merged;
				max_merged = max_merged<locsrt_bucket[j].num_elements?locsrt_bucket[j].num_elements:max_merged;
				sum_merged += locsrt_bucket[j].num_elements;
			}
		}
		printf(" -> %25s: %7u\n", "Num. merged:", num_merged);
		printf(" -> %25s: %7u\n", "Min. keys/merged bucket:", min_merged);
		printf(" -> %25s: %7u\n", "Max. keys/merged bucket:", max_merged);
		printf(" -> %25s: %7u\n", "Avg. keys/merged bucket:", locsort_cfg_counts[i]==0?0:sum_merged/locsort_cfg_counts[i]);
		printf(" -> %25s: %7u\n", "Avg. num. keys:", locsort_cfg_counts[i]==0?0:sum/locsort_cfg_counts[i]);
		printf(" -> Min. bucket size: ");
	}
#endif

	do_compute_locrec_segmented_assignment_offsets<<<1,32, 0, cstrm_extsrt>>>(&(dm->bucket_counters[0].dev_local_bucket_counter[0]), &(dm->bucket_counters[0].dev_local_cfg_offsets[0]));
	do_fast_merged_compute_subbin_offsets_and_segmented_next_pass_assignments<IndexT, RADIX, 8, 128, RDXSRT_MAX_NUM_SORT_CONFIGS, RDXSRT_CFG_MERGE_LOCREC_THRESH, true><<<1, 128, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[0].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[0].dev_local_cfg_offsets[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);
	#else
	do_count_kpb_segmented_locrec_out_bins<RDXSRT_MAX_NUM_SORT_CONFIGS><<<1, RADIX, 0, cstrm_extsrt>>>(dm->dev_histogram, &(dm->bucket_counters[0].dev_local_bucket_counter[0]), dev_locrec_distinctions, num_local_sort_cfgs);
	do_compute_locrec_segmented_assignment_offsets<<<32,1, 0, cstrm_extsrt>>>(&(dm->bucket_counters[0].dev_local_bucket_counter[0]), &(dm->bucket_counters[0].dev_local_cfg_offsets[0]));
	do_compute_subbin_offsets_and_segmented_next_pass_assignments<RDXSRT_MAX_NUM_SORT_CONFIGS><<<1, RADIX, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[0].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[0].dev_local_cfg_offsets[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);
	#endif

	cudaEventRecord(cstrm_nxtp_assignment_compl[0], cstrm_extsrt);
	//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, pfx_sum, pfx_sum, cstrm_extsrt, 0, 8);
	/********* COMPUTE OFFSETS OF OUT-BINS PASS 0 AND ASSIGNMENTS OF PASS 1 *********/
	/********* PERFORM ACTUAL PASS 0 SORTING *********/
	//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, scatter, scatter, cstrm_extsrt, 0, 8);
	if(NUM_PASSES==1){
		// Needs to twiddle in-out in a single pass
		// TODO
	}else{
		rdxsrt_partition_keys<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, true, false><<<num_blocks, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, NULL, 0, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, dev_sorted_keys_out, dev_sorted_values_out);
		if(remaining_elements > 0){
			rdxsrt_partition_keys_with_guards<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, true, false><<<remainder_blocks, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, NULL, 0, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, key_count, num_blocks, dev_sorted_keys_out, dev_sorted_values_out);
		}
//		rdxsrt_partition_keys_with_guards<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, true, false><<<num_blocks+remainder_blocks, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, NULL, 0, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, key_count, 0, dev_sorted_keys_out, dev_sorted_values_out);
	}
	cudaEventRecord(cstrm_lastp_nlsort_compl[0], cstrm_extsrt);
	//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, scatter, scatter, cstrm_extsrt, 0, 8);
	/********* PERFORM ACTUAL PASS 0 SORTING *********/

	unsigned int num_current_pass_ext_bins;
	KeyT *dev_final_sorted_keys_out = (NUM_PASSES%2)==0 ? dev_keys : dev_sorted_keys_out;
	ValueT *dev_final_sorted_values_out = (NUM_PASSES%2)==0 ? dev_values : dev_sorted_values_out;
	/********************
	 * 					*
	 * PASS/BYTE 1,2,.. *
	 *  				*
	 ********************/
	for(int pass = 1; pass < NUM_PASSES; pass++){
		bool IS_VAL_PASS = (pass>=NUM_KEY_PASSES);
		int PASS_BYTE = IS_VAL_PASS ? pass-NUM_KEY_PASSES+sizeof(ValueT)-NUM_LSB_IN_VALUE : pass;

//		printf(" -- Sorting pass %2d, sorts on %s (%d-th most-significant byte) --\n", pass, IS_VAL_PASS?"VALUE":"KEY", PASS_BYTE);

		// Previous pass sorted keys to dev_sorted_keys_out => Now used as input, writing to dev_sorted_keys_out again (Pointer swap)
		std::swap<KeyT*>(dev_keys, dev_sorted_keys_out);
		std::swap<ValueT*>(dev_values, dev_sorted_values_out);

		// Information on the EXTSRT-OUT-bins from previous pass are expected to be at dm->dev_nl_subbucket_info (now renamed to dm->dev_nl_bucket_info)
		std::swap<SubBucketInfo*>(dm->dev_nl_bucket_info, dm->dev_nl_subbucket_info);

		// Prepare EXTSRT block assignments for this pass based on the assignments that have been computed in previous pass
		struct extsrt_assignment_info_t tmp = generate_next_pass_block_assignments(&dm->bucket_counters[pass-1].dev_nl_bucket_counter, dm->dev_nl_bucket_info, dm->nl_bucket_info_buffer, dm->nl_block_assignments, KPB, cstrm_prep_nxtp, cstrm_nxtp_assignment_compl[pass-1]);
		num_current_pass_ext_bins = tmp.num_ext_bins;

		/********* LOCREC SORT PASS N *********/
		unsigned int num_locrec_blocks[RDXSRT_MAX_NUM_SORT_CONFIGS];
//		cudaEventSynchronize(cstrm_nxtp_assignment_compl[pass-1]); IF NEXT_PASS_BLOCK_ASSIGNMENT COMPUTATION IS MOVED BEYOND LOCAL-SORT, WE NEED TO SYNC HERE
		cudaMemcpyAsync(&num_locrec_blocks[0], &(dm->bucket_counters[pass-1].dev_local_bucket_counter[0]), sizeof(num_locrec_blocks), cudaMemcpyDeviceToHost, cstrm_lsort);
		cudaStreamSynchronize(cstrm_lsort);
		cudaStreamWaitEvent(cstrm_lsort, cstrm_lastp_nlsort_compl[pass-1], 0);
		//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, local_sort, local_sort, cstrm_lsort, pass, 8);
		unsigned int locrec_assnmt_offset = 0;
		for(int i=0; i < num_local_sort_cfgs; i++){
			// Skip local sort configurations that won't work on any blocks
			if(num_locrec_blocks[i]<=0)
				continue;
			// Get kernel pointer for the current local sort configuration
			LocalSortKernel<KeyT, ValueT> kernel_ptr = local_sort_configurations->sort_configs[i]->GetSortKernel((NUM_PASSES-pass)*DIGIT_BITS);
			if(!kernel_ptr){
				//--/printf("Couldn't get kernel for config. EXIT.\n");
				exit(-1);
			}
			kernel_ptr<<<num_locrec_blocks[i], local_sort_configurations->sort_configs[i]->tpb, 0, cstrm_lsort>>>(dev_keys, dev_values, dm->dev_local_bucket_info + locrec_assnmt_offset, pass, dev_final_sorted_keys_out, dev_final_sorted_values_out);
			locrec_assnmt_offset += num_locrec_blocks[i];
		}
		//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, local_sort, local_sort, cstrm_lsort, pass, 8);
		cudaEventRecord(cstrm_local_sort_compl[pass], cstrm_lsort);
		/********* LOCREC SORT PASS N *********/

//		/********** PRINT PASS METRICS **********/
//		printf("\n --- PASS %d PREPARATION ---\n", pass);
//		for(int i=0; i<num_local_sort_cfgs; i++)
//			printf(" -> LOCREC-IN-bins[%d]: %8d\n", i, num_locrec_blocks[i]);
//		printf(" -> EXTSRT-IN-bins:       %8d\n", num_current_pass_ext_bins);
//		printf(" -> EXTSRT-FULL-blocks:   %8d\n", tmp.num_extsrt_block_count);
//		printf(" -> EXTSRT-RMNDR-blocks:  %8d\n", tmp.num_extsrt_remainder_block_count);
//		printf(" --- PASS %d PREPARATION ---\n\n", pass);
//		/********** PRINT PASS METRICS **********/

		/********* EXT SORT PASS N *********/
		if(tmp.num_extsrt_block_count + tmp.num_extsrt_remainder_block_count > 0){
			if(IS_VAL_PASS){
				SWAPPER<(NUM_LSB_IN_VAL_PASSES>0 && (cub::Equals<KeyT, ValueT>::VALUE))>::key_lsb_in_value_pointer_swap(dev_keys, dev_values);
				SWAPPER<(NUM_LSB_IN_VAL_PASSES>0 && (cub::Equals<KeyT, ValueT>::VALUE))>::key_lsb_in_value_pointer_swap(dev_sorted_keys_out, dev_sorted_values_out);
			}
			/********* HISTOGRAM PASS N *********/
			// Copy block assignments that have been prepared on the CPU
			cudaMemcpyAsync(dm->dev_nl_block_assignments, dm->nl_block_assignments, (tmp.num_extsrt_block_count+tmp.extsrt_full_block_offset) * sizeof(*dm->dev_nl_block_assignments), cudaMemcpyHostToDevice, cstrm_extsrt);
			//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, histo, histo, cstrm_extsrt, pass, 8);
			rdxsrt_prepare_histograms(dm->dev_histogram, num_current_pass_ext_bins, cstrm_extsrt);
			if(tmp.num_extsrt_block_count > 0)
				rdxsrt_histogram<KeyT, IndexT, DIGIT_BITS, KPT, TPB, 9, false><<<tmp.num_extsrt_block_count, TPB, 0, cstrm_extsrt>>>(dev_keys, dm->dev_nl_block_assignments+tmp.extsrt_full_block_offset, PASS_BYTE, dm->dev_histogram, dm->dev_per_block_histogram);
			if(tmp.num_extsrt_remainder_block_count > 0){
				rdxsrt_histogram_with_guards<KeyT, IndexT, DIGIT_BITS, KPT, TPB, 9, false><<<tmp.num_extsrt_remainder_block_count, TPB, 0, cstrm_extsrt>>>(dev_keys, dm->dev_nl_block_assignments, PASS_BYTE, dm->dev_histogram, dm->dev_per_block_histogram, key_count, tmp.num_extsrt_block_count);
			}
			//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, histo, histo, cstrm_extsrt, pass, 8);
			/********* HISTOGRAM PASS M *********/

			/********* COMPUTE OFFSETS OF OUT-BINS PASS N AND ASSIGNMENTS OF PASS N+1 *********/
			// Compute current pass out-bin offsets and next pass' block assignments
			if(pass == NUM_PASSES-1){
				//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, pfx_sum, pfx_sum, cstrm_extsrt, pass, 8);
				do_fast_merged_compute_subbin_offsets_and_segmented_next_pass_assignments<IndexT, RADIX, 8, 128, RDXSRT_MAX_NUM_SORT_CONFIGS, RDXSRT_CFG_MERGE_LOCREC_THRESH, false><<<num_current_pass_ext_bins, 128, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[pass].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[pass].dev_local_cfg_offsets[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);
				//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, pfx_sum, pfx_sum, cstrm_extsrt, pass, 8);
			}else{
				cudaStreamWaitEvent(cstrm_extsrt, cstrm_local_sort_compl[pass], 0);
				//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, pfx_sum, pfx_sum, cstrm_extsrt, pass, 8);
//				cudaStreamSynchronize(cstrm_lsort);
				#if RDXSRT_CFG_MERGE_LOCREC_THRESH
				do_fast_merged_compute_subbin_offsets_and_segmented_next_pass_assignments<IndexT, RADIX, 8, 128, RDXSRT_MAX_NUM_SORT_CONFIGS, RDXSRT_CFG_MERGE_LOCREC_THRESH, true, true><<<num_current_pass_ext_bins, 128, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[pass].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[pass].dev_local_bucket_counter[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);
				do_compute_locrec_segmented_assignment_offsets<<<1,32, 0, cstrm_extsrt>>>(&(dm->bucket_counters[pass].dev_local_bucket_counter[0]), &(dm->bucket_counters[pass].dev_local_cfg_offsets[0]));
				do_fast_merged_compute_subbin_offsets_and_segmented_next_pass_assignments<IndexT, RADIX, 8, 128, RDXSRT_MAX_NUM_SORT_CONFIGS, RDXSRT_CFG_MERGE_LOCREC_THRESH><<<num_current_pass_ext_bins, 128, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[pass].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[pass].dev_local_cfg_offsets[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);
				#else
				do_count_kpb_segmented_locrec_out_bins<RDXSRT_MAX_NUM_SORT_CONFIGS><<<num_current_pass_ext_bins, RADIX, 0, cstrm_extsrt>>>	(dm->dev_histogram, &(dm->bucket_counters[pass].dev_local_bucket_counter[0]), dev_locrec_distinctions, num_local_sort_cfgs);
				do_compute_locrec_segmented_assignment_offsets<<<32,1, 0, cstrm_extsrt>>>(&(dm->bucket_counters[pass].dev_local_bucket_counter[0]), &(dm->bucket_counters[pass].dev_local_cfg_offsets[0]));
				do_compute_subbin_offsets_and_segmented_next_pass_assignments<RDXSRT_MAX_NUM_SORT_CONFIGS><<<num_current_pass_ext_bins, RADIX, 0, cstrm_extsrt>>>(dm->dev_histogram, dm->dev_nl_bucket_info, dm->dev_subbucket_offsets, &dm->bucket_counters[pass].dev_nl_bucket_counter, dm->dev_nl_subbucket_info, &(dm->bucket_counters[pass].dev_local_cfg_offsets[0]), dm->dev_local_bucket_info, dev_locrec_distinctions, num_local_sort_cfgs);
				#endif
				//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, pfx_sum, pfx_sum, cstrm_extsrt, pass, 8);
			}
			cudaEventRecord(cstrm_nxtp_assignment_compl[pass], cstrm_extsrt);
			/********* COMPUTE OFFSETS OF OUT-BINS PASS N AND ASSIGNMENTS OF PASS N+1 *********/

			/********* PERFORM ACTUAL PASS N SORTING *********/

			//BM_START_CUDA_ARRAY_EVENT_LEV(DBG_LEV, scatter, scatter, cstrm_extsrt, pass, 8);
			if(tmp.num_extsrt_block_count > 0)
				if(pass == NUM_PASSES-1)
					rdxsrt_partition_keys<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, false, true><<<tmp.num_extsrt_block_count, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, dm->dev_nl_block_assignments+tmp.extsrt_full_block_offset, PASS_BYTE, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, dev_sorted_keys_out, dev_sorted_values_out);
				else
					rdxsrt_partition_keys<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, false, false><<<tmp.num_extsrt_block_count, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, dm->dev_nl_block_assignments+tmp.extsrt_full_block_offset, PASS_BYTE, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, dev_sorted_keys_out, dev_sorted_values_out);
			if(tmp.num_extsrt_remainder_block_count > 0)
				if(pass == NUM_PASSES-1)
					rdxsrt_partition_keys_with_guards<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, false, true><<<tmp.num_extsrt_remainder_block_count, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, dm->dev_nl_block_assignments, PASS_BYTE, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, key_count, tmp.num_extsrt_block_count, dev_sorted_keys_out, dev_sorted_values_out);
				else
					rdxsrt_partition_keys_with_guards<KeyT, ValueT, IndexT, DIGIT_BITS, KPT, TPB, 9, false, false><<<tmp.num_extsrt_remainder_block_count, TPB, 0, cstrm_extsrt>>>(dev_keys, dev_values, dm->dev_nl_block_assignments, PASS_BYTE, dm->dev_subbucket_offsets, dm->dev_per_block_histogram, key_count, tmp.num_extsrt_block_count, dev_sorted_keys_out, dev_sorted_values_out);
			//BM_STOP_CUDA_ARRAY_EVENT_LEV(DBG_LEV, scatter, scatter, cstrm_extsrt, pass, 8);
			cudaEventRecord(cstrm_lastp_nlsort_compl[pass], cstrm_extsrt);

			if(IS_VAL_PASS){
				SWAPPER<(NUM_LSB_IN_VAL_PASSES>0 && (cub::Equals<KeyT, ValueT>::VALUE))>::key_lsb_in_value_pointer_swap(dev_keys, dev_values);
				SWAPPER<(NUM_LSB_IN_VAL_PASSES>0 && (cub::Equals<KeyT, ValueT>::VALUE))>::key_lsb_in_value_pointer_swap(dev_sorted_keys_out, dev_sorted_values_out);
			}
			/********* PERFORM ACTUAL PASS N SORTING *********/
		}else{
			// If there were no EXTSRT blocks in this pass, LOCREC should've finished the job already
			break;
		}
	}


	cudaStreamSynchronize(cstrm_extsrt);
	cudaStreamSynchronize(cstrm_prep_nxtp);
	cudaStreamSynchronize(cstrm_lsort);
	//BM_STOP_CUDA_EVENT(sort processing, sort_proc, cstrm_extsrt);
	//BM_STOP_CPU_PROFILE(sort configs, sort_configs);

	if(own_extstrt_stream)
		cudaStreamDestroy(cstrm_extsrt);
	cudaStreamDestroy(cstrm_prep_nxtp);
	for(int i=0; i < NUM_PASSES; i++)
		cudaEventDestroy(cstrm_nxtp_assignment_compl[i]);

	// Destruct default allocated
	if(!pre_allocated_dm)
		delete dm;

	struct RDXSRT_SortedSequence<KeyT, ValueT> ret = {dev_final_sorted_keys_out, dev_final_sorted_values_out};
	return ret;
}

// HOST-SIDE INTERFACE
template <typename KeyT>
void rdxsrt_unstable_sort_keys(KeyT *keys, const unsigned long long int key_count, KeyT *sorted_keys_out)
{
	/*** KEY ALLOCATION ***/
	//BM_START_CUDA_EVENT(memcpy H2D, memcpyhtd,NULL);
	KeyT *dev_keys, *dev_keys_out;
	// Allocate device memory for the sorted keys
	cudaMalloc((void **)&dev_keys_out, sizeof(*dev_keys_out) * key_count);
	// Allocate device memory for the input keys and copy keys there
	cudaMalloc((void **)&dev_keys, sizeof(*dev_keys) * key_count);
	cudaMemcpy(dev_keys, keys, sizeof(*dev_keys) * key_count, cudaMemcpyHostToDevice);
	//BM_STOP_CUDA_EVENT(memcpy H2D, memcpyhtd, NULL);

	/*** Trigger key sorting according to the most significant byte ***/
	//START_CUDA_TIMER(1, bm_whole, NULL);
	//BM_START_CUDA_EVENT(full sort, full_sort, NULL);
//	if(key_count>UINT_MAX)
//		rdxsrt_unstable_sort<KeyT, KeyT, unsigned long long int>(dev_keys, NULL, key_count, dev_keys_out, NULL);
//	else
		rdxsrt_unstable_sort<KeyT, cub::NullType, unsigned int>(dev_keys, NULL, key_count, dev_keys_out, NULL);
	//BM_STOP_CUDA_EVENT(full sort, full_sort, NULL);
	//STOP_CUDA_TIMER(bm_whole, "Whole radix sort...");

	/*** Copy back sorted keys ***/
	//BM_START_CUDA_EVENT(memcpy D2H, memcpydth,NULL);
	cudaMemcpy(sorted_keys_out, dev_keys, sizeof(*dev_keys) * key_count, cudaMemcpyDeviceToHost);
	//BM_STOP_CUDA_EVENT(memcpy D2H, memcpydth,NULL);

	/*** Free memory ***/
	cudaFree(dev_keys);
	cudaFree(dev_keys_out);
}

template <typename KeyT, typename ValueT>
void rdxsrt_unstable_sort_pairs(KeyT *keys, ValueT *values, const unsigned long long int key_count, KeyT *sorted_keys_out, ValueT *sorted_values_out, LocalSortConfigSet<KeyT, ValueT> *local_sort_config_set = NULL)
{
	// Start performance measurement
	/*** KEY ALLOCATION ***/
	//BM_START_CUDA_EVENT(memcpy H2D, memcpyhtd,NULL);
	KeyT *dev_keys, *dev_keys_out;
	// Allocate device memory for the sorted keys
	cudaMalloc((void **)&dev_keys_out, sizeof(*dev_keys_out) * key_count);
	// Allocate device memory for the input keys and copy keys there
	cudaMalloc((void **)&dev_keys, sizeof(*dev_keys) * key_count);
	cudaMemcpy(dev_keys, keys, sizeof(*dev_keys) * key_count, cudaMemcpyHostToDevice);
	/*** KEY ALLOCATION ***/

	/*** VALUE ALLOCATION ***/
	ValueT *dev_values, *dev_values_out;
	// Allocate device memory for the sorted keys
	cudaMalloc((void **)&dev_values_out, sizeof(*dev_values_out) * key_count);
	// Allocate device memory for the input keys and copy keys there
	cudaMalloc((void **)&dev_values, sizeof(*dev_values) * key_count);
	cudaMemcpy(dev_values, values, sizeof(*dev_values) * key_count, cudaMemcpyHostToDevice);
	//BM_STOP_CUDA_EVENT(memcpy H2D, memcpyhtd, NULL);
	/*** VALUE ALLOCATION ***/

	/*** Trigger key sorting according to the most significant byte ***/
	//START_CUDA_TIMER(1, bm_whole, NULL);
	//BM_START_CUDA_EVENT(full sort, full_sort, NULL);
	rdxsrt_unstable_sort<KeyT, ValueT, unsigned int>(dev_keys, dev_values, key_count, dev_keys_out, dev_values_out, local_sort_config_set);
	//BM_STOP_CUDA_EVENT(full sort, full_sort, NULL);
	//STOP_CUDA_TIMER(bm_whole, "Whole radix sort...");

	/*** Copy back sorted keys ***/
	//BM_START_CUDA_EVENT(memcpy D2H, memcpydth,NULL);
	cudaMemcpy(sorted_keys_out, dev_keys, sizeof(*dev_keys) * key_count, cudaMemcpyDeviceToHost);

	/*** Copy back sorted values ***/
	cudaMemcpy(sorted_values_out, dev_values, sizeof(*dev_values) * key_count, cudaMemcpyDeviceToHost);
	//BM_STOP_CUDA_EVENT(memcpy D2H, memcpydth,NULL);

	/*** Free memory ***/
	cudaFree(dev_keys);
	cudaFree(dev_keys_out);
	cudaFree(dev_values);
	cudaFree(dev_values_out);
}

#endif /* GPU_RADIX_SORT_H_ */

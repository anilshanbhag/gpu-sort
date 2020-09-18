#ifndef CUDA_RADIX_SORT_H_
#define CUDA_RADIX_SORT_H_

#include <limits>
#include "cub/cub.cuh"
#include "sort/cuda_radix_sort_common.h"
#include "sort/sorting_network.cuh"
#include "sort/cuda_radix_sort_config.h"

using namespace cub;

__global__ void do_compute_locrec_segmented_assignment_offsets(unsigned int *locrec_per_kpb_atomic_counter_array, unsigned int *locrec_per_kpb_assnmt_offsets);

// __global__ void do_compute_locrec_segmented_assignment_offsets(unsigned int *locrec_per_kpb_atomic_counter_array, unsigned int *locrec_per_kpb_assnmt_offsets)
// {
// 	unsigned int locrec_bin_assinmt_offsets[RDXSRT_MAX_NUM_SORT_CONFIGS];
// 	if(threadIdx.x == 0){
// 		#pragma unroll
// 		for(int i= 0; i < RDXSRT_MAX_NUM_SORT_CONFIGS; i++){
// 			locrec_bin_assinmt_offsets[i] = locrec_per_kpb_atomic_counter_array[i];
// //			printf(" Local Sort Config #%2d: %5d sub-buckets\n", i, locrec_bin_assinmt_offsets[i]);
// 		}

// 		unsigned int tmp_pfx_sum = 0;
// 		unsigned int tmp = 0;
// 		#pragma unroll
// 		for(int i= 0; i < RDXSRT_MAX_NUM_SORT_CONFIGS; i++){
// 			tmp = locrec_bin_assinmt_offsets[i];
// 			locrec_bin_assinmt_offsets[i] = tmp_pfx_sum;
// 			tmp_pfx_sum += tmp;
// 		}

// 		#pragma unroll
// 		for(int i= 0; i < RDXSRT_MAX_NUM_SORT_CONFIGS; i++)
// 			locrec_per_kpb_assnmt_offsets[i] = locrec_bin_assinmt_offsets[i];
// 	}
// }

/***********************************
 * HELPER FUNCTIONS
 ***********************************/

enum RDXSRT_TWIDDLE_MODE {
	RDXSRT_TWIDDLE_MODE_NONE,
	RDXSRT_TWIDDLE_MODE_IN,
	RDXSRT_TWIDDLE_MODE_OUT,
	RDXSRT_TWIDDLE_MODE_INOUT,
};

template<
	typename 	KeyT,	 					// Type of key data being loaded from device memory
	typename 	UnsignedKeyT,	 			// Unsigned integer type of the shared memory holding the buckets
	typename 	IndexT,	 					// Index/offset type used
	int 		NUM_BITS,					// Number of bits being sorted at a time
	int			KPT,						// Number of keys per thread
	int			TPB,						// Number of threads per block (used for strided loading)
	enum RDXSRT_LOAD_STRIDE LOAD_STRIDE,	// Stride method being used for a thread's keys
	bool		SORT_RUN,					// Whether or not to pre-sort the runs to maximise likelihood of look-ahead hits
	int 		LOOK_AHEAD,					// The number of tuples to be considered at once (1+how_far_to_look_ahead)
	int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
	int 		NUM_ITEMS, 					// Iteration count (number of items to fetch, starting from the given offset)
	bool		GUARD_NUM_ITEMS,			// If the thread block's total number of items isn't exactly equal to KPT*TPB and block_item_count shall be respected
	enum RDXSRT_TWIDDLE_MODE BIT_TWIDDLE_MODE
>
struct LocalPartitioningRun
{
	__device__ __forceinline__ void LocalPartitionKeysFromMemory(IndexT block_base_offset, IndexT thread_id, KeyT *keys_in, const unsigned int digit, UnsignedKeyT (&keys)[KPT], unsigned int (&keys_masked)[KPT], unsigned int *shared_block_local_offsets, UnsignedKeyT *shared_block_local_sorted, unsigned int block_item_count)
	{
		/*** Typedefs ***/
		typedef Traits<KeyT>                        	KeyTraits;
		typedef typename KeyTraits::UnsignedBits    	UnsignedBits;
		typedef LoadUnit<IndexT, LOAD_STRIDE, KPT, TPB>	KeyLoader;

		/*** GET KEYS ***/
		if(GUARD_NUM_ITEMS){
			KeyLoader(block_base_offset, thread_id).template LoadStridedWithGuards<UnsignedBits, KeyT, ITERATOR_OFFSET, NUM_ITEMS>(keys_in, keys, block_item_count);
		}else{
			KeyLoader(block_base_offset, thread_id).template LoadStrided<UnsignedBits, KeyT, ITERATOR_OFFSET, NUM_ITEMS>(keys_in, keys);
		}


		/*** BIT TWIDDLING ***/
		if(BIT_TWIDDLE_MODE == RDXSRT_TWIDDLE_MODE_IN || BIT_TWIDDLE_MODE == RDXSRT_TWIDDLE_MODE_INOUT){
			#pragma unroll
			for(int i=0; i<NUM_ITEMS; i++){
				keys[ITERATOR_OFFSET+i] = KeyTraits::TwiddleIn(keys[ITERATOR_OFFSET+i]);
			}
		}

		/*** SORT RUN ***/
		if(SORT_RUN){
			SortingNetwork<UnsignedKeyT>::sort<NUM_ITEMS>(&keys[ITERATOR_OFFSET]);
		}

		/*** MASK KEYS FOR FAST LOOK-UPs ***/
		#pragma unroll
		for(int i=0; i<NUM_ITEMS;i++){
			keys_masked[ITERATOR_OFFSET+i] = (keys[ITERATOR_OFFSET+i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
		}

		/*** BIT TWIDDLING ***/
		if(BIT_TWIDDLE_MODE == RDXSRT_TWIDDLE_MODE_OUT || BIT_TWIDDLE_MODE == RDXSRT_TWIDDLE_MODE_INOUT){
			#pragma unroll
			for(int i=0; i<NUM_ITEMS; i++){
				keys[ITERATOR_OFFSET+i] = KeyTraits::TwiddleOut(keys[ITERATOR_OFFSET+i]);
			}
		}

		/*** PARTITION KEYS INTO SHARED MEMORY BUCKETS ***/
		IndexT tmp_offset;
		int is_inserted = 0;
		#pragma unroll
		for(int i=0; i<NUM_ITEMS-(LOOK_AHEAD-1); i++){
			if((!GUARD_NUM_ITEMS) || (KeyLoader(block_base_offset, thread_id).ThreadIndexInBounds(block_item_count, ITERATOR_OFFSET+i))){
				// Element already inserted (only check if look ahead is used)
				if(LOOK_AHEAD>1 && is_inserted>0){
					is_inserted--;
					keys_masked[ITERATOR_OFFSET+i] = keys_masked[ITERATOR_OFFSET+i-1] + 1;
				}
				// Current element still needs to be inserted: is_inserted==0
				else{
					// Look-ahead
					#pragma unroll
					for(int la=1;la<LOOK_AHEAD;la++)
						if(keys_masked[ITERATOR_OFFSET+i] == keys_masked[ITERATOR_OFFSET+i+la])
							is_inserted = la;
						else
							break;
					tmp_offset = atomicAdd(&shared_block_local_offsets[keys_masked[ITERATOR_OFFSET+i]], 1+is_inserted);
					keys_masked[ITERATOR_OFFSET+i] = tmp_offset;

					if(LOOK_AHEAD<=1){
						shared_block_local_sorted[tmp_offset] = keys[ITERATOR_OFFSET+i];
					}else{
						// Insertion (including look-ahead insertion)
						#pragma unroll
						for(int la=0;la<LOOK_AHEAD;la++)
							if(la < is_inserted+1)
								shared_block_local_sorted[tmp_offset+la] = keys[ITERATOR_OFFSET+i+la];
					}
				}
			}
		}

		#pragma unroll
		for(int i=0; i<LOOK_AHEAD-1; i++){
			if(is_inserted > 0){
				is_inserted--;
				keys_masked[ITERATOR_OFFSET+NUM_ITEMS-LOOK_AHEAD+1+i] = keys_masked[ITERATOR_OFFSET+NUM_ITEMS-LOOK_AHEAD+1+i-1] + 1;
			}else{
				tmp_offset = atomicAdd(&shared_block_local_offsets[keys_masked[ITERATOR_OFFSET+NUM_ITEMS-LOOK_AHEAD+1+i]], 1);
				keys_masked[ITERATOR_OFFSET+NUM_ITEMS-LOOK_AHEAD+1+i] = tmp_offset;
				shared_block_local_sorted[tmp_offset] = keys[ITERATOR_OFFSET+NUM_ITEMS-LOOK_AHEAD+1+i];
			}
		}
	}
};

template<bool ADD_BUCKET_TO_QUEUE, int THREADS_PER_BUCKET>
struct AppendBigBucket{
	__device__ __forceinline__ void write_small_buckets(const unsigned int thread_id, unsigned int *big_bucket_queue, unsigned int bucket_idx);
};
template<int THREADS_PER_BUCKET>
struct AppendBigBucket<false, THREADS_PER_BUCKET>{
	__device__ __forceinline__ void write_small_buckets(const unsigned int thread_id, unsigned int *big_bucket_queue, unsigned int bucket_idx){};
};

template<int THREADS_PER_BUCKET>
struct AppendBigBucket<true, THREADS_PER_BUCKET>{
	__device__ __forceinline__ void write_small_buckets(const unsigned int thread_id, unsigned int *big_bucket_queue, unsigned int bucket_idx){
		if(thread_id%THREADS_PER_BUCKET == 0){
			unsigned int offset = atomicAdd(&big_bucket_queue[0], 1);
			big_bucket_queue[1+offset] = bucket_idx;
		}
	};
};

template<
typename 		KeyT,	 					// Type of key data being loaded from device memory
typename		IndexT,
unsigned int	TPB,
unsigned int	NUM_BUCKETS,
unsigned int 	THREADS_PER_BUCKET,
bool 			READ_ONLY_BIG_BUCKETS,
unsigned int 	BIG_BUCKET_THRESH,
unsigned int 	THREADS_PER_BIG_BUCKET
>
__device__ __forceinline__ void write_small_buckets(const unsigned int thread_id, KeyT *items_out, const IndexT *__restrict__ items_out_bucket_offsets, const KeyT *__restrict__ shared_items_in, const unsigned int (&items_in_bucket_sizes)[NUM_BUCKETS], const unsigned int (&items_in_bucket_offsets)[NUM_BUCKETS], unsigned int *big_bucket_queue)
{
	unsigned int bucket_offset = thread_id/THREADS_PER_BUCKET;

	/*** ITERATING OVER THE BUCKETS ***/
	#pragma unroll
	for(int i=0;i<NUM_BUCKETS/(TPB/THREADS_PER_BUCKET);i++){
		/*** BIG BUCKETS***/
		if(items_in_bucket_sizes[bucket_offset] > BIG_BUCKET_THRESH){
			AppendBigBucket<!READ_ONLY_BIG_BUCKETS, THREADS_PER_BUCKET>().write_small_buckets(thread_id, big_bucket_queue, bucket_offset);
		}
		/*** SMALL BUCKETS ***/
		else{
			#pragma unroll 1
			for(unsigned int idx = thread_id%THREADS_PER_BUCKET; idx < items_in_bucket_sizes[bucket_offset]; idx += THREADS_PER_BUCKET){
				// Write to where this bin starts <PLUS> beginning of the sub-bin's offset reserved for this block <PLUS> the index the element has within this block
				items_out[items_out_bucket_offsets[bucket_offset] + idx] = shared_items_in[items_in_bucket_offsets[bucket_offset] + idx];
			}
		}
		bucket_offset+=(TPB/THREADS_PER_BUCKET);
	}

	/*** ITERATING OVER THE LAST BUCKET (IF NECCESSARY) ***/
	if(bucket_offset<NUM_BUCKETS){
		/*** BIG BUCKETS***/
		if(items_in_bucket_sizes[bucket_offset] > BIG_BUCKET_THRESH){
			AppendBigBucket<!READ_ONLY_BIG_BUCKETS, THREADS_PER_BUCKET>().write_small_buckets(thread_id, big_bucket_queue, bucket_offset);
		}
		/*** SMALL BUCKETS ***/
		else{
			#pragma unroll 1
			for(unsigned int idx = thread_id%THREADS_PER_BUCKET; idx < items_in_bucket_sizes[bucket_offset]; idx += THREADS_PER_BUCKET){
				// Write to where this bin starts <PLUS> beginning of the sub-bin's offset reserved for this block <PLUS> the index the element has within this block
				items_out[items_out_bucket_offsets[bucket_offset] + idx] = shared_items_in[items_in_bucket_offsets[bucket_offset] + idx];
			}
		}
	}
}

template<
typename 		KeyT,	 					// Type of key data being loaded from device memory
typename		IndexT,
unsigned int	TPB,
unsigned int	NUM_BUCKETS,
unsigned int 	THREADS_PER_BUCKET
>
__device__ __forceinline__ void write_huge_buckets(const unsigned int thread_id, KeyT *items_out, const IndexT *__restrict__ items_out_bucket_offsets, const KeyT *__restrict__ shared_items_in, const unsigned int (&items_in_bucket_sizes)[NUM_BUCKETS], const unsigned int (&items_in_bucket_offsets)[NUM_BUCKETS], unsigned int *big_bucket_queue)
{
	unsigned int num_big_buckets = *big_bucket_queue;
	// Iterate over the buckets
	for(int i=thread_id/THREADS_PER_BUCKET; i<num_big_buckets; i+=TPB/THREADS_PER_BUCKET){
		unsigned int big_bucket_idx = big_bucket_queue[i+1];
		#pragma unroll 10
		for(unsigned int idx = thread_id%THREADS_PER_BUCKET; idx < items_in_bucket_sizes[big_bucket_idx]; idx += THREADS_PER_BUCKET){
			// Write to where this bin starts <PLUS> beginning of the sub-bin's offset reserved for this block <PLUS> the index the element has within this block
			items_out[items_out_bucket_offsets[big_bucket_idx] + idx] = shared_items_in[items_in_bucket_offsets[big_bucket_idx] + idx];
		}
	}
}

template<
typename 		KeyT,	 					// Type of key data being loaded from device memory
typename		IndexT,
unsigned int	TPB,
unsigned int	NUM_BUCKETS,
unsigned int 	THREADS_PER_BUCKET,
bool 			READ_ONLY_BIG_BUCKETS,
unsigned int 	BIG_BUCKET_THRESH,
unsigned int 	THREADS_PER_BIG_BUCKET
>
__device__ __forceinline__ void write_buckets(const unsigned int thread_id, KeyT *items_out, const IndexT *__restrict__ items_out_bucket_offsets, const KeyT *__restrict__ shared_items_in, const unsigned int (&items_in_bucket_sizes)[NUM_BUCKETS], const unsigned int (&items_in_bucket_offsets)[NUM_BUCKETS], unsigned int *big_bucket_queue)
{
	/*** SMALL BUCKETS ***/
	// Enough threads to have several full thread groups
	if(TPB>THREADS_PER_BUCKET){
		if((TPB%THREADS_PER_BUCKET == 0) || (thread_id < TPB-(TPB%THREADS_PER_BUCKET))){
			write_small_buckets<KeyT, IndexT, TPB-(TPB%THREADS_PER_BUCKET), NUM_BUCKETS, THREADS_PER_BUCKET, READ_ONLY_BIG_BUCKETS, BIG_BUCKET_THRESH, THREADS_PER_BIG_BUCKET>(thread_id, items_out, items_out_bucket_offsets, shared_items_in, items_in_bucket_sizes, items_in_bucket_offsets, big_bucket_queue);
		}
	}
	// Not even one full thread group => Make it 1 thread group with all TPB being assigned to it
	else{
		write_small_buckets<KeyT, IndexT, TPB, NUM_BUCKETS, TPB, READ_ONLY_BIG_BUCKETS, BIG_BUCKET_THRESH, THREADS_PER_BIG_BUCKET>(thread_id, items_out, items_out_bucket_offsets, shared_items_in, items_in_bucket_sizes, items_in_bucket_offsets, big_bucket_queue);
	}

	// Sorting the very large buckets
//	if(!READ_ONLY_BIG_BUCKETS)
	__syncthreads();

	/*** BIG BUCKETS ***/
	if(TPB>THREADS_PER_BIG_BUCKET){
		if((TPB%THREADS_PER_BIG_BUCKET == 0) || (thread_id < TPB-(TPB%THREADS_PER_BIG_BUCKET))){
			write_huge_buckets<KeyT, IndexT, TPB-(TPB%THREADS_PER_BIG_BUCKET), NUM_BUCKETS, THREADS_PER_BIG_BUCKET>(thread_id, items_out, items_out_bucket_offsets, shared_items_in, items_in_bucket_sizes, items_in_bucket_offsets, big_bucket_queue);
		}
	}else{
		write_huge_buckets<KeyT, IndexT, TPB, NUM_BUCKETS, TPB>(thread_id, items_out, items_out_bucket_offsets, shared_items_in, items_in_bucket_sizes, items_in_bucket_offsets, big_bucket_queue);
	}
}

template<
typename 	KeyT,	 					// Type of key data being loaded from device memory
typename 	UnsignedKeyT,	 			// Unsigned integer type of the shared memory holding the buckets
typename 	IndexT,	 					// Index/offset type used
int 		NUM_BITS,					// Number of bits being sorted at a time
int			KPT,						// Number of keys per thread
int			TPB,						// Number of threads per block (used for strided loading)
enum RDXSRT_LOAD_STRIDE LOAD_STRIDE,	// Stride method being used for a thread's keys
bool		SORT_RUN,					// Whether or not to pre-sort the runs to maximise likelihood of look-ahead hits
int 		LOOK_AHEAD,					// The number of tuples to be considered at once (1+how_far_to_look_ahead)
int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
int 		NUM_ITEMS, 					// Iteration count (number of items to fetch, starting from the given offset)
bool		GUARD_NUM_ITEMS,			// If the thread block's total number of items isn't exactly equal to KPT*TPB and block_item_count shall be respected
bool 		IS_LAST_RUN,
enum RDXSRT_TWIDDLE_MODE BIT_TWIDDLE_MODE
>
struct LocalPartitioningRunExecutor;


template<
typename 	KeyT,	 					// Type of key data being loaded from device memory
typename 	UnsignedKeyT,	 			// Unsigned integer type of the shared memory holding the buckets
typename 	IndexT,	 					// Index/offset type used
int 		NUM_BITS,					// Number of bits being sorted at a time
int			KPT,						// Number of keys per thread
int			TPB,						// Number of threads per block (used for strided loading)
enum RDXSRT_LOAD_STRIDE LOAD_STRIDE,	// Stride method being used for a thread's keys
bool		SORT_RUN,					// Whether or not to pre-sort the runs to maximise likelihood of look-ahead hits
int 		LOOK_AHEAD,					// The number of tuples to be considered at once (1+how_far_to_look_ahead)
int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
int 		NUM_ITEMS, 					// Iteration count (number of items to fetch, starting from the given offset)
bool		GUARD_NUM_ITEMS,			// If the thread block's total number of items isn't exactly equal to KPT*TPB and block_item_count shall be respected
enum RDXSRT_TWIDDLE_MODE BIT_TWIDDLE_MODE
>
struct LocalPartitioningRunExecutor<KeyT, UnsignedKeyT, IndexT, NUM_BITS, KPT, TPB, LOAD_STRIDE, SORT_RUN, LOOK_AHEAD, ITERATOR_OFFSET, NUM_ITEMS, GUARD_NUM_ITEMS, true, BIT_TWIDDLE_MODE>
{
	__device__ __forceinline__ void local_partitioning(IndexT block_base_offset, IndexT thread_id, KeyT *keys_in, const unsigned int digit, UnsignedKeyT (&keys)[KPT], unsigned int (&keys_masked)[KPT], unsigned int *shared_block_local_offsets, UnsignedKeyT *shared_block_local_sorted, unsigned int block_item_count)
	{
		// Last run, look ahead larger or equal to number of remaining items => don't use look-ahead
		if(NUM_ITEMS<LOOK_AHEAD){
			(LocalPartitioningRun<KeyT, UnsignedKeyT, IndexT, NUM_BITS, KPT, TPB, LOAD_STRIDE, SORT_RUN, 1, ITERATOR_OFFSET, (ITERATOR_OFFSET+NUM_ITEMS>KPT?KPT:NUM_ITEMS), GUARD_NUM_ITEMS, BIT_TWIDDLE_MODE>()).LocalPartitionKeysFromMemory(block_base_offset, thread_id, keys_in, digit, keys, keys_masked, shared_block_local_offsets, shared_block_local_sorted, block_item_count);
		}
		// Last run, still enough items for reasonable use of look-ahead
		else{
			(LocalPartitioningRun<KeyT, UnsignedKeyT, IndexT, NUM_BITS, KPT, TPB, LOAD_STRIDE, SORT_RUN, LOOK_AHEAD, ITERATOR_OFFSET, (ITERATOR_OFFSET+NUM_ITEMS>KPT?KPT:NUM_ITEMS), GUARD_NUM_ITEMS, BIT_TWIDDLE_MODE>()).LocalPartitionKeysFromMemory(block_base_offset, thread_id, keys_in, digit, keys, keys_masked, shared_block_local_offsets, shared_block_local_sorted, block_item_count);
		}
	}
};

#define MIN_HELPER(A,B) ((A)<(B)?(A):(B))
template<
typename 	KeyT,	 					// Type of key data being loaded from device memory
typename 	UnsignedKeyT,	 			// Unsigned integer type of the shared memory holding the buckets
typename 	IndexT,	 					// Index/offset type used
int 		NUM_BITS,					// Number of bits being sorted at a time
int			KPT,						// Number of keys per thread
int			TPB,						// Number of threads per block (used for strided loading)
enum RDXSRT_LOAD_STRIDE LOAD_STRIDE,	// Stride method being used for a thread's keys
bool		SORT_RUN,					// Whether or not to pre-sort the runs to maximise likelihood of look-ahead hits
int 		LOOK_AHEAD,					// The number of tuples to be considered at once (1+how_far_to_look_ahead)
int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
int 		NUM_ITEMS, 					// Iteration count (number of items to fetch, starting from the given offset)
bool		GUARD_NUM_ITEMS,				// If the thread block's total number of items isn't exactly equal to KPT*TPB and block_item_count shall be respected
enum RDXSRT_TWIDDLE_MODE BIT_TWIDDLE_MODE
>
struct LocalPartitioningRunExecutor<KeyT, UnsignedKeyT, IndexT, NUM_BITS, KPT, TPB, LOAD_STRIDE, SORT_RUN, LOOK_AHEAD, ITERATOR_OFFSET, NUM_ITEMS, GUARD_NUM_ITEMS, false, BIT_TWIDDLE_MODE>
{
	__device__ __forceinline__ void local_partitioning(IndexT block_base_offset, IndexT thread_id, KeyT *keys_in, const unsigned int digit, UnsignedKeyT (&keys)[KPT], unsigned int (&keys_masked)[KPT], unsigned int *shared_block_local_offsets, UnsignedKeyT *shared_block_local_sorted, unsigned int block_item_count)
	{
		(LocalPartitioningRun<KeyT, UnsignedKeyT, IndexT, NUM_BITS, KPT, TPB, LOAD_STRIDE, SORT_RUN, LOOK_AHEAD, ITERATOR_OFFSET, NUM_ITEMS, GUARD_NUM_ITEMS, BIT_TWIDDLE_MODE>()).LocalPartitionKeysFromMemory(block_base_offset, thread_id, keys_in, digit, keys, keys_masked, shared_block_local_offsets, shared_block_local_sorted, block_item_count);
		// NUM_ITEMS OR => MIN( NUM_ITEMS, NUM_ITEMS<KPT-OFST ) KPT-OFFSET
		(LocalPartitioningRunExecutor<KeyT, UnsignedKeyT, IndexT, NUM_BITS, KPT, TPB, LOAD_STRIDE, SORT_RUN, LOOK_AHEAD, ITERATOR_OFFSET+NUM_ITEMS, MIN_HELPER(NUM_ITEMS,KPT-(ITERATOR_OFFSET+NUM_ITEMS)), GUARD_NUM_ITEMS, (NUM_ITEMS>=KPT-(ITERATOR_OFFSET+NUM_ITEMS)), BIT_TWIDDLE_MODE>()).local_partitioning(block_base_offset, thread_id, keys_in, digit, keys, keys_masked, shared_block_local_offsets, shared_block_local_sorted, block_item_count);
	}
};



template <
	typename KeyT, 				// Data type of the keys within device memory. Bits will be twiddled (if it's the MSD) to unsigned type, and back to KeyT if it's the last pass
	typename ValueT, 			// Data type of the values associated with the keys
	typename IndexT, 			// Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32-1)
	int NUM_BITS, 				// Number of bits being sorted at a time
	int KPT, 					// Number of keys per thread
	int TPB, 					// Number of threads per block
	int RUN_LENGTH,				//
	bool IS_MSD,				// Whether or not this pass is the most-significant digit.
	bool IS_LAST_PASS			// Whether or not this pass is the last pass, involving the least-significant bits.
>
__global__ void rdxsrt_partition_keys(KeyT *__restrict__ keys_in, ValueT *__restrict__ values_in, const struct rdxsrt_extsrt_block_to_bin_t *__restrict__ block_to_bin_assignments, const unsigned int digit, IndexT *__restrict__ global_offsets, const unsigned int *__restrict__ per_block_histo, KeyT *__restrict__ keys_out, ValueT *__restrict__ values_out)
{
	// Bucket index used to determine the memory offset of the bucket's global histogram
	const unsigned int bucket_idx = IS_MSD ? 0 : block_to_bin_assignments[blockIdx.x].bin_idx;
	// This thread block's keys memory offset, pointing to the index of its first key
	const IndexT block_offset = IS_MSD ? (blockDim.x * blockIdx.x * KPT) : block_to_bin_assignments[blockIdx.x].offset;
	// The offset of the keys of the bucket to which this thread block is assigned to
	const IndexT bucket_offset = IS_MSD ? 0 : block_to_bin_assignments[blockIdx.x].bin_start_offset;

	/*** TYPEDEFS ***/
	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;
	typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;
	const unsigned int RADIX = 0x01<<NUM_BITS;

	/*** DECLARATIONS ***/
	unsigned int __shared__ block_local_histo[RADIX];			// Histogram over the block's keys
	unsigned int __shared__ block_local_orig_offsets[RADIX];	// Prefix-sum over the block's histogram
	unsigned int __shared__ block_local_offsets[RADIX];			// The current block's sub-bucket's offsets
	IndexT __shared__ shared_global_offset[RADIX];				// The global sub-bucket's offsets for this block's keys
	__shared__ union PartitionTempStorage{
		PartitionTempStorage (){}
		UnsignedBits keys[KPT*TPB];			// Shared memory for distributing keys into respective sub-buckets
		ValueT values[KPT*TPB];
	}block_local_sorted;

	UnsignedBits tloc_keys[KPT];
	unsigned int keys_masked[KPT];

	/*** GET BLOCK'S HISTOGRAM ***/

	const int IPT = (TPB-1+RADIX)/TPB;
	unsigned int threads_histo_counters[IPT];

	#pragma unroll
	for(int i=0;i<IPT;i++){
		if(i + IPT*threadIdx.x < RADIX){
			threads_histo_counters[i] = per_block_histo[blockIdx.x * RADIX + i + IPT*threadIdx.x];
			block_local_histo[i + IPT*threadIdx.x] = threads_histo_counters[i];
			shared_global_offset[i + IPT*threadIdx.x] = bucket_offset + atomicAdd(&global_offsets[bucket_idx*RADIX + i + IPT*threadIdx.x], threads_histo_counters[i]);
		}else{
			threads_histo_counters[i] = 0;
		}
	}

	unsigned int max_i;
	unsigned int tmp_max_i;

	/*** COMPUTE PREFIX-SUM OVER HISTO & FIND MAX SUB-BUCKET ***/
	// Make sure we're invoking it with a warp-multiple number of threads
	ExclusivePrefixSumAndMax<unsigned int, true, IPT, ((RDXSRT_WARP_THREADS*IPT-1+RADIX)/(RDXSRT_WARP_THREADS*IPT)) *RDXSRT_WARP_THREADS, TPB, RADIX>(threads_histo_counters, block_local_offsets, max_i, tmp_max_i);

	// Copy the prefix-sum results
	#pragma unroll
	for(int i=0;i<IPT;i++){
			if(i + IPT*threadIdx.x < RADIX){
			block_local_orig_offsets[i + IPT*threadIdx.x] = block_local_offsets[i + IPT*threadIdx.x];
		}
	}

	// Make sure that block_local_orig_offsets were copied before modifying block_local_offsets during the sort processe
	__syncthreads();

	/*** HOT BUCKET CASE ***/
	if(/*false && */(Equals<ValueT, cub::NullType>::VALUE) && max_i > KPT*TPB/3){
		const enum RDXSRT_TWIDDLE_MODE TWIDDLE_MODE = (IS_MSD&&IS_LAST_PASS) ? RDXSRT_TWIDDLE_MODE_INOUT : (IS_MSD ? RDXSRT_TWIDDLE_MODE_IN : (IS_LAST_PASS ? RDXSRT_TWIDDLE_MODE_OUT : RDXSRT_TWIDDLE_MODE_NONE));
		(LocalPartitioningRunExecutor<KeyT, UnsignedBits, unsigned int, NUM_BITS, KPT, TPB, RDXSRT_LOAD_STRIDE_WARP, true, 3, 0, RUN_LENGTH, false, (RUN_LENGTH>=KPT), TWIDDLE_MODE>()).local_partitioning(block_offset, threadIdx.x, keys_in, digit, tloc_keys, keys_masked, block_local_offsets, block_local_sorted.keys, KPT*TPB);
	}
	/*** NO HOT BUCKET CASE ***/
	else{
		/*** PARTITION KEYS INTO SHARED MEMORY BUCKETS ***/
		const enum RDXSRT_TWIDDLE_MODE TWIDDLE_MODE = (IS_MSD&&IS_LAST_PASS) ? RDXSRT_TWIDDLE_MODE_INOUT : (IS_MSD ? RDXSRT_TWIDDLE_MODE_IN : (IS_LAST_PASS ? RDXSRT_TWIDDLE_MODE_OUT : RDXSRT_TWIDDLE_MODE_NONE));
		(LocalPartitioningRunExecutor<KeyT, UnsignedBits, unsigned int, NUM_BITS, KPT, TPB, RDXSRT_LOAD_STRIDE_WARP, false, 1, 0, RUN_LENGTH, false, (RUN_LENGTH>=KPT), TWIDDLE_MODE>()).local_partitioning(block_offset, threadIdx.x, keys_in, digit, tloc_keys, keys_masked, block_local_offsets, block_local_sorted.keys, KPT*TPB);
	}
	/*** WRITE OUT THE RESULTS ***/
	// Make sure block local sort is in shared memory
	__syncthreads();

	/*** WRITE OUT ***/
	if(threadIdx.x == 0)
		block_local_offsets[0] = 0;
	__syncthreads();

	write_buckets<KeyT, IndexT, TPB, RADIX, 32, false, 1024, 128>(threadIdx.x, keys_out, shared_global_offset, reinterpret_cast<KeyT*>(block_local_sorted.keys), block_local_histo, block_local_orig_offsets, block_local_offsets);
	/*** END WRITE-OUT***/

	/**************
	 * VALUES
	 **************/
	if(Equals<ValueT, cub::NullType>::VALUE)
		return;

		__syncthreads();

	/*** GET KEYS ***/
		ValueT values[KPT];
		KeyLoader(block_offset, threadIdx.x).template LoadStrided<ValueT, ValueT, 0, KPT>(values_in, values);

		#pragma unroll
		for(int i=0; i<KPT; i++){
			block_local_sorted.values[keys_masked[i]] = values[i];
		}
		__syncthreads();

		write_buckets<ValueT, IndexT, TPB, RADIX, 32, true, 1024, 128>(threadIdx.x, values_out, shared_global_offset, block_local_sorted.values, block_local_histo, block_local_orig_offsets, block_local_offsets);
}


template <
	typename KeyT, 				// Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
	typename ValueT, 			// Data type of the values associated with the keys within device memory.
	typename IndexT, 			// Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
	int NUM_BITS, 				// Number of bits being sorted at a time
	int KPT, 					// Number of keys per thread
	int TPB, 					// Number of threads per block
	int RUN_LENGTH,				//
	bool IS_MSD,				// Whether or not this is the histogram on the most-significant digit
	bool IS_LAST_PASS
>
__global__ void rdxsrt_partition_keys_with_guards(KeyT *__restrict__ keys_in, ValueT *__restrict__ values_in, const struct rdxsrt_extsrt_block_to_bin_t *__restrict__ block_to_bin_assignments, const unsigned int digit, IndexT *__restrict__ global_offsets, const unsigned int *__restrict__ per_block_histo, const unsigned int total_keys, const int block_index_offset, KeyT *__restrict__ keys_out, ValueT *__restrict__ values_out)
{
	// Bucket index used to determine the memory offset of the bucket's global histogram
	const unsigned int bucket_idx = IS_MSD ? 0 : block_to_bin_assignments[blockIdx.x].bin_idx;
	// This thread block's keys memory offset, pointing to the index of its first key
	const IndexT block_offset = IS_MSD ? (blockDim.x * (blockIdx.x+block_index_offset) * KPT) : block_to_bin_assignments[blockIdx.x].offset;
	// The offset of the keys of the bucket to which this thread block is assigned to
	const IndexT bucket_offset = IS_MSD ? 0 : block_to_bin_assignments[blockIdx.x].bin_start_offset;

	const IndexT block_num_keys = IS_MSD ? total_keys : block_to_bin_assignments[blockIdx.x].block_num_elements + block_offset;

	/*** TYPEDEFS ***/
	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;
	typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;
	const unsigned int RADIX = 0x01<<NUM_BITS;

	/*** DECLARATIONS ***/
	unsigned int __shared__ block_local_histo[RADIX];			// Histogram over the block's keys
	unsigned int __shared__ block_local_orig_offsets[RADIX];	// Prefix-sum over the block's histogram
	unsigned int __shared__ block_local_offsets[RADIX];		// The current block's sub-bucket's offsets
	IndexT __shared__ shared_global_offset[RADIX];				// The global sub-bucket's offsets for this block's keys

	__shared__ union PartitionTempStorage{
		PartitionTempStorage(){}
		UnsignedBits keys[KPT*TPB];			// Shared memory for distributing keys into respective sub-buckets
		ValueT values[KPT*TPB];
	}block_local_sorted;

	UnsignedBits tloc_keys[KPT];
	unsigned int keys_masked[KPT];

	/*** GET BLOCK'S HISTOGRAM ***/

	const int IPT = (TPB-1+RADIX)/TPB;
	unsigned int threads_histo_counters[IPT];

	#pragma unroll
	for(int i=0;i<IPT;i++){
		if(i + IPT*threadIdx.x < RADIX){
			threads_histo_counters[i] = per_block_histo[(blockIdx.x+block_index_offset) * RADIX + i + IPT*threadIdx.x];
			block_local_histo[i + IPT*threadIdx.x] = threads_histo_counters[i];
			shared_global_offset[i + IPT*threadIdx.x] = bucket_offset + atomicAdd(&global_offsets[bucket_idx*RADIX + i + IPT*threadIdx.x], threads_histo_counters[i]);
		}else{
			threads_histo_counters[i] = 0;
		}
	}

	unsigned int max_i;
	unsigned int tmp_max_i;

	/*** COMPUTE PREFIX-SUM OVER HISTO & FIND MAX SUB-BUCKET ***/
	// Make sure we're invoking it with a warp-multiple number of threads
	ExclusivePrefixSumAndMax<unsigned int, true, IPT, ((RDXSRT_WARP_THREADS*IPT-1+RADIX)/(RDXSRT_WARP_THREADS*IPT)) *RDXSRT_WARP_THREADS, TPB, RADIX>(threads_histo_counters, block_local_offsets, max_i, tmp_max_i);

	// Copy the prefix-sum results
	#pragma unroll
	for(int i=0;i<IPT;i++){
		if(i + IPT*threadIdx.x < RADIX){
			block_local_orig_offsets[i + IPT*threadIdx.x] = block_local_offsets[i + IPT*threadIdx.x];
		}
	}

	// Make sure that block_local_orig_offsets were copied before modifying block_local_offsets during the sort processe
	__syncthreads();

	/*** HOT BUCKET CASE ***/
	if(false && max_i > KPT*TPB/3){
#if false
		(LocalPartitioningRunExecutor<KeyT, UnsignedBits, unsigned int, NUM_BITS, KPT, TPB, RDXSRT_LOAD_STRIDE_WARP, !IS_MSD, 3, 0, RUN_LENGTH, (RUN_LENGTH>=KPT)>()).local_partitioning(block_offset, threadIdx.x, keys_in, digit, tloc_keys, keys_masked, block_local_offsets, block_local_sorted);

		// Make sure block local sort is in shared memory
		__syncthreads();

		/*** WRITE OUT ***/
		if(threadIdx.x == 0)
			block_local_offsets[0] = 0;
		__syncthreads();

		// Every warp works on 32 bins
		#pragma unroll 21
		for(int bucket=threadIdx.x/32; bucket < RADIX; bucket+=TPB/32){
			if(block_local_histo[bucket] > 1024){
				if(threadIdx.x%32 == 0){
					unsigned int offset = atomicAdd(&block_local_offsets[0], 1);
					block_local_offsets[1+offset] = bucket;
				}
			}else{
				#pragma unroll 1
				for(unsigned int idx = threadIdx.x%32; idx < block_local_histo[bucket]; idx += 32){
					// Write to where this bin starts <PLUS> beginning of the sub-bin's offset reserved for this block <PLUS> the index the element has within this block
					keys_out[bucket_offset + shared_global_offset[bucket] + idx] = block_local_sorted[block_local_orig_offsets[bucket] + idx];
				}
			}
		}

		// Sorting the very large buckets
		__syncthreads();
		for(int i=threadIdx.x/128; i<block_local_offsets[0]; i+=TPB/128){
			tmp_max_i = block_local_offsets[1+i];
			#pragma unroll 10
			for(unsigned int idx = threadIdx.x%128; idx < block_local_histo[tmp_max_i]; idx += 128){
				// Write to where this bin starts <PLUS> beginning of the sub-bin's offset reserved for this block <PLUS> the index the element has within this block
				keys_out[bucket_offset+ shared_global_offset[tmp_max_i] + idx] = block_local_sorted[block_local_orig_offsets[tmp_max_i] + idx];
			}
		}
		/*** END WRITE-OUT***/
#endif
	}
	/*** NO HOT BUCKET CASE ***/
	else{
		/*** PARTITION KEYS INTO SHARED MEMORY BUCKETS ***/
		const enum RDXSRT_TWIDDLE_MODE TWIDDLE_MODE = (IS_MSD&&IS_LAST_PASS) ? RDXSRT_TWIDDLE_MODE_INOUT : (IS_MSD ? RDXSRT_TWIDDLE_MODE_IN : (IS_LAST_PASS ? RDXSRT_TWIDDLE_MODE_OUT : RDXSRT_TWIDDLE_MODE_NONE));
		(LocalPartitioningRunExecutor<KeyT, UnsignedBits, unsigned int, NUM_BITS, KPT, TPB, RDXSRT_LOAD_STRIDE_WARP, false, 1, 0, RUN_LENGTH, true, (RUN_LENGTH>=KPT), TWIDDLE_MODE>()).local_partitioning(block_offset, threadIdx.x, keys_in, digit, tloc_keys, keys_masked, block_local_offsets, block_local_sorted.keys, block_num_keys);

		// Make sure block local sort is in shared memory
		__syncthreads();

		/*** WRITE OUT ***/
		if(threadIdx.x == 0)
			block_local_offsets[0] = 0;
		__syncthreads();

		// Every warp works on 32 bins
		write_buckets<KeyT, IndexT, TPB, RADIX, 32, false, 1024, 128>(threadIdx.x, keys_out, shared_global_offset, reinterpret_cast<KeyT*>(block_local_sorted.keys), block_local_histo, block_local_orig_offsets, block_local_offsets);
		/*** END WRITE-OUT***/

		/**************
		 * VALUES
		 **************/
		if(Equals<ValueT, cub::NullType>::VALUE)
			return;

		__syncthreads();

		/*** GET KEYS ***/
		ValueT values[KPT];
		KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<ValueT, ValueT, 0, KPT>(values_in, values, block_num_keys);

		#pragma unroll
		for(int i=0; i<KPT; i++){
			if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_num_keys, i))
				block_local_sorted.values[keys_masked[i]] = values[i];
		}
		__syncthreads();

		write_buckets<ValueT, IndexT, TPB, RADIX, 32, true, 1024, 128>(threadIdx.x, values_out, shared_global_offset, block_local_sorted.values, block_local_histo, block_local_orig_offsets, block_local_offsets);
	}
}


/***********************************
 *
 ***********************************/
/**
 * Computes the histogram over the digit values of an array of keys that MUST have a length of an integer multiple of (KPT * blockDim.x).
 * The padding to the integer multiple can be done by adding 0's at the end and subtracting the number of padded 0's from the final result's 0 bin.
 * The 2^NUM_BITS possible counts (0..2^NUM_BITSNUM_BITS-1) will be placed in global_histo.
 * @param keys						[IN] 	The keys for which to compute the histogram
 * @param block_to_bin_assignments	[IN]
 * @param digit						[IN]
 * @param global_histo				[OUT]	The array of element counts, MUST be 256 in size.
 * @param per_block_histo			[OUT]
 */
template<
	typename KeyT,				// Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
	typename IndexT,			// Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
	int NUM_BITS,				// Number of bits being sorted at a time
	int KPT,					// Number of keys per thread
	int TPB,					// Number of threads per block
	int PRE_SORT_RUNS_LENGTH,	// For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
	bool IS_MSD					// Whether or not this is the histogram on the most-significant digit
>
__global__ void rdxsrt_histogram(KeyT *__restrict__ keys, const struct rdxsrt_extsrt_block_to_bin_t *__restrict__ block_to_bin_assignments, const unsigned int digit, IndexT *global_histo, unsigned int *per_block_histo)
{
	/*** TYPEDEFs***/
	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;
	typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;

	/*** DECLARATIONS ***/
	UnsignedBits tloc_keys[KPT];
	unsigned int tloc_masked[KPT];
	__shared__ unsigned int shared_bins[0x01<<NUM_BITS];

	/*** INIT SHARED HISTO ***/
	if(threadIdx.x < 32){
		#pragma unroll
		for(int i=0;i<(0x01<<NUM_BITS);i+=32){
			shared_bins[i+threadIdx.x] = 0;
		}
	}
	__syncthreads();

	/*** GET KEYS & PREPARE KEYS FOR HISTO ***/
	// Bucket index used to determine the memory offset of the bucket's global histogram
	const unsigned int bucket_idx = IS_MSD ? 0 : block_to_bin_assignments[(blockIdx.x)].bin_idx;
	// This thread block's keys memory offset, pointing to the index of its first key
	const IndexT block_offset = IS_MSD ? (blockDim.x * blockIdx.x * KPT) : block_to_bin_assignments[(blockIdx.x)].offset;

	// Load keys
	KeyLoader(block_offset, threadIdx.x).template LoadStrided<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys);
#if true || USE_RLE_HISTO
	// Mask
	#pragma unroll
	for(int i=0; i<KPT; i++){
		if(IS_MSD)
			tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
		tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
	}

	/*** SORT RUNS ***/
	if(PRE_SORT_RUNS_LENGTH>1){
		SortingNetwork<unsigned int>::sort_runs<PRE_SORT_RUNS_LENGTH>(tloc_masked);

	}

	/*** COMPUTE HISTO ***/
	unsigned int rle = 1;
	#pragma unroll
	for(int i=1; i<KPT; i++){
		if(tloc_masked[i] == tloc_masked[i-1])
			rle++;
		else{
			atomicAdd(&shared_bins[tloc_masked[i-1]], rle);
			rle=1;
		}
	}
	atomicAdd(&shared_bins[tloc_masked[KPT-1]], rle);
#else
	#pragma unroll
	for(int i=0; i<KPT; i++){
		tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
		atomicAdd(&shared_bins[tloc_masked[i]], 1);
	}
#endif

	// Make sure we've got the counts from all threads
	__syncthreads();

	/*** Write shared histo to global histo ***/
	if(threadIdx.x < 32){
		for(int i=0;i<(0x01<<NUM_BITS);i+=32){
			atomicAdd(&global_histo[(0x01<<NUM_BITS)*bucket_idx+i+threadIdx.x], shared_bins[i+threadIdx.x]);
			per_block_histo[blockIdx.x*(0x01<<NUM_BITS)+i+threadIdx.x] = shared_bins[i+threadIdx.x];
		}
	}
}


template<
	typename KeyT,				// Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
	typename IndexT,			// Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
	int NUM_BITS,				// Number of bits being sorted at a time
	int KPT,					// Number of keys per thread
	int TPB,					// Number of threads per block
	int PRE_SORT_RUNS_LENGTH,	// For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
	bool IS_MSD				// Whether or not this is the histogram on the most-significant digit
>
__global__ void rdxsrt_histogram_with_guards(KeyT *__restrict__ keys, const struct rdxsrt_extsrt_block_to_bin_t *__restrict__ block_to_bin_assignments, const unsigned int digit, IndexT *global_histo, unsigned int *per_block_histo, const IndexT total_keys, const int block_index_offset)
{
	/*** TYPEDEFs***/
	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;
	typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;

	/*** DECLARATIONS ***/
	UnsignedBits tloc_keys[KPT];
	unsigned int tloc_masked[KPT];
	__shared__ unsigned int shared_bins[(0x01<<NUM_BITS) + 1];

	/*** INIT SHARED HISTO ***/
	if(threadIdx.x < 32){
		#pragma unroll
		for(int i=0;i<(0x01<<NUM_BITS);i+=32){
			shared_bins[i+threadIdx.x] = 0;
		}
	}
	__syncthreads();

	/*** GET KEYS & PREPARE KEYS FOR HISTO ***/
	// Bucket index used to determine the memory offset of the bucket's global histogram
	const unsigned int bucket_idx = IS_MSD ? 0 : block_to_bin_assignments[(blockIdx.x)].bin_idx;
	// This thread block's keys memory offset, pointing to the index of its first key
	const IndexT block_offset = IS_MSD ? (blockDim.x * (block_index_offset + blockIdx.x) * KPT) : block_to_bin_assignments[(blockIdx.x)].offset;

	// Maximum number of keys the block may fetch
	const IndexT block_max_num_keys = IS_MSD ? total_keys : block_to_bin_assignments[(blockIdx.x)].block_num_elements + block_offset;
	KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys, block_max_num_keys);
	#pragma unroll
	for(int i=0; i<KPT; i++){
		if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_num_keys, i)){
			if(IS_MSD)
				tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
			tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
			atomicAdd(&shared_bins[tloc_masked[i]], 1);
		}
	}

	// Make sure we've got the counts from all threads
	__syncthreads();

	/*** Write shared histo to global histo ***/
	if(threadIdx.x < 32){
		for(int i=0;i<(0x01<<NUM_BITS);i+=32){
			atomicAdd(&global_histo[(0x01<<NUM_BITS)*bucket_idx+i+threadIdx.x], shared_bins[i+threadIdx.x]);
			per_block_histo[(block_index_offset + blockIdx.x)*(0x01<<NUM_BITS)+i+threadIdx.x] = shared_bins[i+threadIdx.x];
		}
	}
}



template <
	int NUM_DISTINCTIONS,
	int MERGE_THRESHOLD,
	unsigned int MAX_MERGE_THRESHOLD
>
__global__ void do_merged_count_kpb_segmented_locrec_out_bins(
		const unsigned int *__restrict__ in_bucket_histo,			// [IN]  The buckets' histograms (one histogram per thread block)
		unsigned int *locsort_per_distinction_counter,				// [OUT] Counts the number of sub-buckets that are going to be sorted by a local sort with the given threshold
		const int *__restrict__ distinction_thresholds,				// [IN]  The sub-bucket size distinctions for the local sort
		const int num_distinction_thresholds						// [IN]  Total number of sub-bucket size distinctions for the local sort
		)
{
	__shared__ unsigned int subbin_counts[256];
	unsigned int tloc_count = in_bucket_histo[blockIdx.x*256+threadIdx.x];
	subbin_counts[threadIdx.x] = tloc_count;
	__syncthreads();

	unsigned int locrec_bin_counters[NUM_DISTINCTIONS];
	#pragma unroll
	for(int i = 0; i < NUM_DISTINCTIONS; i++)
		locrec_bin_counters[i] = 0;

	/*** SINGLE THREADED PROCESSING ***/
	if(threadIdx.x == 0){
		int b = 0;
		while(b < 256){
			tloc_count = subbin_counts[b];
			if(tloc_count>0){

				// Is locrec
				if(tloc_count<=distinction_thresholds[num_distinction_thresholds-1]){

					/*** MERGE ***/
					// As long as an additional bucket is below MERGE_THRESHOLD
					if(tloc_count<MERGE_THRESHOLD)
						while(b+1 < 256 && tloc_count + subbin_counts[b+1] < MAX_MERGE_THRESHOLD){
							tloc_count += subbin_counts[b+1];
							b++;
						}

					/*** IDENTIFY THE KERNEL THAT SHALL BE USED FOR THAT BUCKET ***/
					for(int i=0; i < NUM_DISTINCTIONS; i++){
						if(tloc_count <= distinction_thresholds[i]){
							locrec_bin_counters[i]++;
							break;
						}
					}
				}
			}
			b++;
		}
		for(int i= 0; i<num_distinction_thresholds; i++){
			unsigned int c = atomicAdd(&locsort_per_distinction_counter[i], locrec_bin_counters[i]);
//			printf("Locrec Count %2d: %3d%", i, c+locsort_per_distinction_counter[i]);
		}

	}
}

template <int NUM_DISTINCTIONS>
__global__ void do_count_kpb_segmented_locrec_out_bins(
		const unsigned int *__restrict__ in_bucket_histo,
		unsigned int *locsort_per_distinction_counter,
		const int *__restrict__ distinction_thresholds,
		const int num_distinction_thresholds
		)
{
	unsigned int tloc_count = in_bucket_histo[blockIdx.x*256+threadIdx.x];

	unsigned int locrec_bin_counters[NUM_DISTINCTIONS];
	#pragma unroll
	for(int i = 0; i < NUM_DISTINCTIONS; i++)
		locrec_bin_counters[i] = 0;

	// Every thread of this ext-IN-bin's offset computation now checks if its OUT-bin is sorted LOCREC in the next pass
	if(tloc_count>0 && tloc_count <= distinction_thresholds[num_distinction_thresholds-1]){
		locrec_bin_counters[0] += tloc_count <= distinction_thresholds[0];
		#pragma unroll
		for(int i=1; i < NUM_DISTINCTIONS; i++){
			locrec_bin_counters[i] += (i<num_distinction_thresholds) && (tloc_count > distinction_thresholds[i-1]) && (tloc_count <= distinction_thresholds[i]);
		}
	}

	for(int i= 0; i<num_distinction_thresholds; i++)
		atomicAdd(&locsort_per_distinction_counter[i], locrec_bin_counters[i]);
}


template <
	int NUM_DISTINCTIONS,
	unsigned int MERGE_THRESHOLD,
	unsigned int MAX_MERGE_THRESHOLD
>
__global__ void do_merged_compute_subbin_offsets_and_segmented_next_pass_assignments(
		const unsigned int *__restrict__ global_histo,								// [IN]  The bucket's histogram
		SubBucketInfo *__restrict__ block_to_bin_assignments,				// [IN]  Information on the bucket (offset)
		unsigned int *__restrict__ sub_bucket_offsets,									// [OUT] Information for the absolute sub-bucket's offsets
		unsigned int *atomic_nl_subbucket_counter,									// [OUT]
		SubBucketInfo *__restrict__ nl_subbucket_info,
		unsigned int *locrec_per_kpb_assnmt_offsets,
		struct rdxsrt_recsrt_block_info_t<MAX_MERGE_THRESHOLD> *__restrict__ local_sort_block_assignments,
		const int *__restrict__ distinction_thresholds,
		const int num_distinction_thresholds)
{
	__shared__ unsigned int subbin_counts[256];
	__shared__ unsigned int subbin_offsets[256];
	unsigned int tloc_count = global_histo[blockIdx.x*256+threadIdx.x];
	subbin_counts[threadIdx.x] = tloc_count;
	__syncthreads();

	CUDA_SHARED_HISTO_EXCL_PFX_SUM_DECLARATIONS;
	CUDA_SHARED_HISTO_EXCL_PFX_SUM(tloc_count, subbin_offsets, true);

	sub_bucket_offsets[block_to_bin_assignments[blockIdx.x].bin_idx*256+threadIdx.x] = subbin_offsets[threadIdx.x];

	/*** SINGLE THREADED ***/
	if(threadIdx.x == 0){
		int b = 0;
		int current_bin;
		while(b < 256){
			tloc_count = subbin_counts[b];
			if(tloc_count>0){

				// Is locrec
				if(tloc_count<=distinction_thresholds[num_distinction_thresholds-1]){
					current_bin = b;
					bool merged = false;
					unsigned int idx;

					/*** MERGE ***/
					// As long as an additional bucket is below MERGE_THRESHOLD
					if(tloc_count<MERGE_THRESHOLD)
						while(b+1 < 256 && tloc_count + subbin_counts[b+1] < MAX_MERGE_THRESHOLD){
							merged = true;
							tloc_count += subbin_counts[b+1];
							b++;
						}

					// TODO improvement: only compute address of atomic_inc ptr in the conditionals and do adds together without branching
					/*** IDENTIFY THE KERNEL THAT SHALL BE USED FOR THAT BUCKET ***/
					for(int i=0; i < num_distinction_thresholds; i++){
						if(tloc_count <= distinction_thresholds[i]){
							idx = atomicAdd(&locrec_per_kpb_assnmt_offsets[i], 1);
//							printf("RESERVED (cfg %d) sub-bucket assignment added at idx: %u (%p)\n", i, idx, &locrec_per_kpb_assnmt_offsets[i]);
							break;
						}
					}

					/*** INSERT INTO THE RESPECTIVE KERNEL QUEUE ***/
					struct rdxsrt_recsrt_block_info_t<MAX_MERGE_THRESHOLD> tmp;
					// Offset is equal to: offset of the ext-IN-bin PLUS the relative offset of the thread's OUT-bin
					tmp.offset = block_to_bin_assignments[blockIdx.x].offset + subbin_offsets[current_bin];
					tmp.num_elements = tloc_count;
#if RDXSRT_CFG_MERGE_LOCREC_THRESH
					tmp.is_merged = merged ? 1 : 0;
#endif
//					printf("Local sort sub-bucket assignment added at idx: %u (%p-%p)\n", idx, &local_sort_block_assignments[idx], &local_sort_block_assignments[idx]+sizeof(tmp));
					local_sort_block_assignments[idx] = tmp;
					printf("Local, %5d, %3d, %6d, %6u\n",blockIdx.x, 0, tmp.offset, tmp.num_elements);
				}else{
					unsigned int idx = atomicAdd(atomic_nl_subbucket_counter, 1);
					SubBucketInfo tmp;
					tmp.offset = block_to_bin_assignments[blockIdx.x].offset + subbin_offsets[b]; 				// STILL TO DO on CPU
					tmp.block_num_elements = tloc_count; 											// STILL TO PROPERLY DO on CPU
					tmp.bin_idx = idx; 																// TODO figure out if bin index needs to be in order (i.e. leftmost bin has idx 0, ...)
					nl_subbucket_info[idx] = tmp;
				}
			}
			b++;
		}
	}

}



template <
	typename IndexT,
	int RADIX,
	int ITEMS_PER_THREAD,
	int TPB,
	int NUM_DISTINCTIONS,
	unsigned int MERGE_THRESHOLD,
	bool PREPARE_NEXT_PASS_ASSIGNMENTS = true,
	bool COUNT_ONLY = false
>
__global__ void do_fast_merged_compute_subbin_offsets_and_segmented_next_pass_assignments(
		IndexT *__restrict__ global_histo,								// [IN]  The bucket's histogram
		SubBucketInfo *__restrict__ block_to_bin_assignments,			// [IN]  Information on the bucket (offset)
		IndexT *__restrict__ sub_bucket_offsets,						// [OUT] Information for the absolute sub-bucket's offsets
		unsigned int *atomic_nl_subbucket_counter,						// [OUT]
		SubBucketInfo *__restrict__ nl_subbucket_info,
		unsigned int *local_sort_config_num_subbuckets,
		struct rdxsrt_recsrt_block_info_t<MERGE_THRESHOLD> *__restrict__ local_sort_block_assignments,
		const int *__restrict__ distinction_thresholds,
		const int num_distinction_thresholds)
{
	// Loader to load consecutive ITEMS_PER_THREAD counters from the histogram this block is assigned to
	typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_SINGLE, ITEMS_PER_THREAD, TPB> HistogramLoader;

	__shared__ IndexT subbucket_offsets[RADIX];
	__shared__ IndexT subbucket_counter[RADIX];

	/*** LOAD DATA ***/
	IndexT t_histo_counters[ITEMS_PER_THREAD];
	HistogramLoader(blockIdx.x*RADIX, threadIdx.x).template LoadStridedWithGuards<IndexT, IndexT, 0, ITEMS_PER_THREAD>(global_histo, t_histo_counters, blockIdx.x*RADIX+RADIX);

	/*** COMPUTE SUB-BUCKET OFFSETS FOR CURRENT PASS's PARTITIONING STEP ***/
	if(!COUNT_ONLY){
		/*** BACKUP ORIGINAL COUNTS ***/
		#pragma unroll
		for(int i=0; i<ITEMS_PER_THREAD; i++)
			if(threadIdx.x*ITEMS_PER_THREAD+i<RADIX)
				subbucket_counter[threadIdx.x*ITEMS_PER_THREAD+i] = t_histo_counters[i];

		/*** COMPUTE PREFIX-SUM ***/
		const int PFX_SUM_THREADS = ((RDXSRT_WARP_THREADS*ITEMS_PER_THREAD-1+RADIX)/(RDXSRT_WARP_THREADS*ITEMS_PER_THREAD)) *RDXSRT_WARP_THREADS;
		unsigned int max_i;
		unsigned int tmp_max_i;
		ExclusivePrefixSumAndMax<IndexT, false, ITEMS_PER_THREAD, PFX_SUM_THREADS, TPB, RADIX>(t_histo_counters, subbucket_offsets, max_i, tmp_max_i);
//		sub_bucket_offsets[block_to_bin_assignments[blockIdx.x].bin_idx*RADIX + threadIdx.x] = subbucket_offsets[threadIdx.x];

		for(int i=threadIdx.x; i<RADIX; i+=TPB)
			sub_bucket_offsets[block_to_bin_assignments[blockIdx.x].bin_idx*RADIX + i] = subbucket_offsets[i];

		/*** RESTORE ORIGINAL COUNTS ***/
		#pragma unroll
		for(int i=0; i<ITEMS_PER_THREAD; i++)
			if(threadIdx.x*ITEMS_PER_THREAD+i<RADIX)
				t_histo_counters[i] = subbucket_counter[threadIdx.x*ITEMS_PER_THREAD+i];
	}

	/*** PREPARE ASSIGNMENTS ***/
	if(PREPARE_NEXT_PASS_ASSIGNMENTS){
		__syncthreads();

		/***
		 * After each thread finished merging its thread-local sub-buckets (as far as possible), it inserts its leftmost sub-bucket (LEFT)
		 * and its rightmost sub-bucket (RIGHT) into shared memory, iff the respective sub-bucket still runs below the merge-threshold.
		 * Otherwise, num_elements of greater than MERGE_THRESHOLD will be written, to indicate there's no mergeable sub-bucket of the thread.
		 *  - post_merge[2 * threadIdx.x + 0] => LEFT
		 *  - post_merge[2 * threadIdx.x + 1] => RIGHT
		 * ***/
		__shared__ rdxsrt_recsrt_block_info_t<MERGE_THRESHOLD> post_merge[2 * (ITEMS_PER_THREAD-1+RADIX)/ITEMS_PER_THREAD];

		/*** Counts the number of local sub-buckets that have been written to shared memory so far ***/
		__shared__ unsigned int local_sub_bucket_buffer_counter;
		__shared__ rdxsrt_recsrt_block_info_t<MERGE_THRESHOLD> local_sub_bucket_buffer[RADIX];
		__shared__ unsigned int nl_bucket_buffer_counter;
		__shared__ unsigned int nl_bucket_base_offset;
		__shared__ unsigned int nl_bucket_buffer[RADIX];

		if(threadIdx.x == 0){
			local_sub_bucket_buffer_counter = 0;
			nl_bucket_buffer_counter = 0;
		}
		__syncthreads();

		/*** MERGE TINY BUCKETS, DETERMINE LOCAL-SORT-CONFIG FOR SMALL BUCKETS, AND IDENTIFY NON-LOCAL BUCKETS ***/
		rdxsrt_recsrt_block_info_t<MERGE_THRESHOLD> current_sub_bucket;
		#define RDXSRT_BARRIER_NUM_KEYS 999999

		// By default, there's no LEFT- or RIGHT-mergeable sub-buckets: num_elements > Local-Sort-Threshold
		if(HistogramLoader(blockIdx.x*RADIX, threadIdx.x).ThreadIndexInBounds(blockIdx.x*RADIX+RADIX, 0)){
			post_merge[threadIdx.x*2 + 0].num_elements = RDXSRT_BARRIER_NUM_KEYS;
			post_merge[threadIdx.x*2 + 1].num_elements = RDXSRT_BARRIER_NUM_KEYS;
		}

		bool is_left = true;
		current_sub_bucket.set_merged(false);
		current_sub_bucket.num_elements = t_histo_counters[0];
		current_sub_bucket.offset = threadIdx.x*ITEMS_PER_THREAD+0;

		#pragma unroll
		for(int i=1; i<ITEMS_PER_THREAD; i++){
			if(HistogramLoader(blockIdx.x*RADIX, threadIdx.x).ThreadIndexInBounds(blockIdx.x*RADIX+RADIX, i)){

				/*** MERGE TINY SUB-BUCKETS ***/
				if(current_sub_bucket.num_elements + t_histo_counters[i] < MERGE_THRESHOLD){
					current_sub_bucket.num_elements += t_histo_counters[i];
					current_sub_bucket.set_merged(current_sub_bucket.get_merged() || (t_histo_counters[i]>0));
				}

				/*** NO-MERGE ***/
				else{

					/*** EMPTY SUB-BUCKET ***/
					if(current_sub_bucket.num_elements == 0){
						// Skip it, even if it's the left-sub-bucket. Here's why:
						// Merging with a '0' sub-bucket only makes sense to bridge to a right, non-zero sub-bucket of the successor thread.
						// However, in this case, there's no mergeable right bucket.
					}

					/*** SMALL BUCKET (LOCAL SORT) ***/
					else if(current_sub_bucket.num_elements<=distinction_thresholds[num_distinction_thresholds-1]){
						// It's the left sub-bucket, which possibly could be merged with preceding thread's right sub-bucket
						if(is_left && current_sub_bucket.num_elements < MERGE_THRESHOLD){
							post_merge[threadIdx.x * 2 + 0] = current_sub_bucket;
						}else{
							/*** IDENTIFY THE KERNEL THAT SHALL BE USED FOR THAT BUCKET ***/
							unsigned int idx = atomicAdd(&local_sub_bucket_buffer_counter, 1);
							local_sub_bucket_buffer[idx].num_elements = current_sub_bucket.num_elements;
							if(!COUNT_ONLY){
								local_sub_bucket_buffer[idx].offset = current_sub_bucket.offset;
								local_sub_bucket_buffer[idx].set_merged(current_sub_bucket.get_merged());
							}
						}
					}
					/*** NON-LOCAL BUCKET (NON-LOCAL-SORT) ***/
					else{
						if(!COUNT_ONLY){
							unsigned int idx = atomicAdd(&nl_bucket_buffer_counter, 1);
							nl_bucket_buffer[idx] = current_sub_bucket.offset;
						}
					}

					/*** SET THE NEW 'CURRENT' SUB-BUCKET ***/
					current_sub_bucket.offset = threadIdx.x*ITEMS_PER_THREAD+i;
					current_sub_bucket.num_elements = t_histo_counters[i];
					current_sub_bucket.set_merged(false);

					/*** THERE's DEFINITLY NO MORE LEFT BUCKET ***/
					is_left = false;
				}
			}
		}

		/*** MAKE SURE THAT WHAT WE SET INITIALLY WAS ACTUALLY TRUE: AT LEAST, THREAD's FIRST ITEM IS IN BOUNDS ***/
		if(HistogramLoader(blockIdx.x*RADIX, threadIdx.x).ThreadIndexInBounds(blockIdx.x*RADIX+RADIX, 0)){
			// We simply insert a both-way mergeable sub-bucket. Event if it's a '0' counter,
			// as it might help bridge the preceding thread's right with the successor thread's left sub-bucket.
			/*** BOTH-WAY-MERGEABLE ***/
			if(is_left && current_sub_bucket.num_elements < MERGE_THRESHOLD){
				post_merge[threadIdx.x * 2 + 0] = current_sub_bucket;
				post_merge[threadIdx.x * 2 + 1] = current_sub_bucket;
			}else{
				/*** ACCUMULATED SUB-BUCKET HAS NO KEYS ***/
				if(current_sub_bucket.num_elements == 0){
					// Skip it...
				}
				/*** ACCUMULATED SUB-BUCKET IS SMALL BUCKET (LOCAL SORT) ***/
				else if(current_sub_bucket.num_elements<=distinction_thresholds[num_distinction_thresholds-1]){
					/*** RIGHT-WAY-MERGEABLE ***/
					if(current_sub_bucket.num_elements < MERGE_THRESHOLD){
						post_merge[threadIdx.x * 2 + 1] = current_sub_bucket;
					}else{
						unsigned int idx = atomicAdd(&local_sub_bucket_buffer_counter, 1);
						local_sub_bucket_buffer[idx].num_elements = current_sub_bucket.num_elements;
						if(!COUNT_ONLY){
							local_sub_bucket_buffer[idx].offset = current_sub_bucket.offset;
							local_sub_bucket_buffer[idx].set_merged(current_sub_bucket.get_merged());
						}
					}
				}
				/*** ACCUMULATED SUB-BUCKET IS NON-LOCAL BUCKET (NON-LOCAL-SORT) ***/
				else{
					if(!COUNT_ONLY){
						unsigned int idx = atomicAdd(&nl_bucket_buffer_counter, 1);
						nl_bucket_buffer[idx] = current_sub_bucket.offset;
					}
				}
			}
		}
		__syncthreads();


		/*** WE MIGHT HAVE:
		 *  -> Two consecutive buckets at an even index, followed by an uneven index that are representing only one single sub-bucket.
		 *  -> Barrier (current_bucket cannot be merged with bucket at index i) iff:
		 *  	1) --> post_merge[i].num_elements == RDXSRT_BARRIER_NUM_KEYS
		 *  	OR
		 *  	2) --> (i%2==1) && (post_merge[i].offset != post_merge[i-1].offset)
		 *  	OR
		 *  	3) --> current_sub_bucket.num_elements == RDXSRT_BARRIER_NUM_KEYS
		 *  	OR (and this should also cover 3) and 1)!
		 *  	4) --> current_sub_bucket.num_elements + post_merge[i].current_sub_bucket.num_elements > MERGE_THRESHOLD
		 *  -> Mergeable
		 */

		if(threadIdx.x == 0){
			current_sub_bucket = post_merge[0*2 + 0];

			for(int i=1;i < ((ITEMS_PER_THREAD-1+RADIX)/ITEMS_PER_THREAD)*2; i++){
				/*** Bridging Sub-Bucket (two sub-buckets, the thread's LEFT and RIGHT are representing one and the same sub-bucket) ***/
				if((i%2 == 1 && post_merge[i].offset == post_merge[i-1].offset && post_merge[i-1].num_elements == post_merge[i].num_elements )){

				}else{
					/*** NO MORE MERGING ***/
					if(	/*post_merge[i].num_elements == RDXSRT_BARRIER_NUM_KEYS ||
						current_sub_bucket.num_elements == RDXSRT_BARRIER_NUM_KEYS ||*/ // These two are covered by the following conditional (if either of both exceed, then the sum of both exceeds anyway)
						(i%2 == 1) || // Is the right bucket, and, since the top level conditional would've fetched a bridging LEFT==RIGHT sub-bucket case, we can be sure that this is NOT a bridging sub-bucket
						current_sub_bucket.num_elements + post_merge[i].num_elements > MERGE_THRESHOLD
						){

						// If the current bucket is actually an invalid bucket (e.g. was inserted as a barrier)
						if((current_sub_bucket.num_elements == 0) || (current_sub_bucket.num_elements > MERGE_THRESHOLD)){
							// Don't do anything with a symbolic 'barrier'
						}else{
							unsigned int idx = atomicAdd(&local_sub_bucket_buffer_counter, 1);
							local_sub_bucket_buffer[idx].num_elements = current_sub_bucket.num_elements;
							if(!COUNT_ONLY){
								local_sub_bucket_buffer[idx].offset = current_sub_bucket.offset;
								local_sub_bucket_buffer[idx].set_merged(current_sub_bucket.get_merged());
							}
						}

						/*** SET BUCKET AT CURRENT INDEX AS THE NEW 'CURRENT BUCKET' ***/
						current_sub_bucket = post_merge[i];
					}else{
						current_sub_bucket.set_merged(current_sub_bucket.get_merged() || (post_merge[i].num_elements>0));
						current_sub_bucket.num_elements += post_merge[i].num_elements;
					}
				}
			}


			if((current_sub_bucket.num_elements == 0) || (current_sub_bucket.num_elements > MERGE_THRESHOLD)){
				// Don't do anything with a symbolic 'barrier'
			}else{
				unsigned int idx = atomicAdd(&local_sub_bucket_buffer_counter, 1);
				local_sub_bucket_buffer[idx].num_elements = current_sub_bucket.num_elements;
				if(!COUNT_ONLY){
					local_sub_bucket_buffer[idx].offset = current_sub_bucket.offset;
					local_sub_bucket_buffer[idx].set_merged(current_sub_bucket.get_merged());
				}
			}
		}

		__syncthreads();

		/*** WRITE BUFFERED LOCAL SORT BUCKETS ***/
		#pragma unroll 4
		for(int idx=threadIdx.x; idx<local_sub_bucket_buffer_counter;idx+=TPB){
			/*** IDENTIFY THE KERNEL THAT SHALL BE USED FOR THAT BUCKET ***/
			int local_sort_config_idx = 0;
			for(int i=0; i < num_distinction_thresholds; i++){
				if(local_sub_bucket_buffer[idx].num_elements <= distinction_thresholds[i]){
					local_sort_config_idx = i;
					break;
				}
			}
			unsigned int glob_idx = atomicAdd(&local_sort_config_num_subbuckets[local_sort_config_idx], 1);

			/*** INSERT INTO THE RESPECTIVE KERNEL QUEUE ***/
			if(!COUNT_ONLY){
				local_sort_block_assignments[glob_idx].num_elements = local_sub_bucket_buffer[idx].num_elements;
				local_sort_block_assignments[glob_idx].offset 		= block_to_bin_assignments[blockIdx.x].offset + subbucket_offsets[local_sub_bucket_buffer[idx].offset];
				local_sort_block_assignments[glob_idx].set_merged(local_sub_bucket_buffer[idx].get_merged());
			}
		}

		/*** WRITE BUFFERED NON-LOCAL SORT BUCKETS ***/
		if(!COUNT_ONLY){
			if(threadIdx.x == 0){
				nl_bucket_base_offset = atomicAdd(atomic_nl_subbucket_counter, nl_bucket_buffer_counter);
			}
			__syncthreads();

			for(int idx=threadIdx.x; idx<nl_bucket_buffer_counter;idx+=TPB){
				nl_subbucket_info[nl_bucket_base_offset+idx].bin_idx 				= nl_bucket_base_offset + idx;
				nl_subbucket_info[nl_bucket_base_offset+idx].block_num_elements	= subbucket_counter[nl_bucket_buffer[idx]];
				nl_subbucket_info[nl_bucket_base_offset+idx].offset 				= block_to_bin_assignments[blockIdx.x].offset + subbucket_offsets[nl_bucket_buffer[idx]];
			}
		}
	}
}

template <int NUM_DISTINCTIONS>
__global__ void do_compute_subbin_offsets_and_segmented_next_pass_assignments(
		const unsigned int *__restrict__ global_histo,
		SubBucketInfo *__restrict__ in_bin_info,
		unsigned int *__restrict__ global_offsets,
		unsigned int *atomic_block_assignment_counter,
		SubBucketInfo *__restrict__ block_assignments,
		unsigned int *locrec_per_kpb_assnmt_offsets,
		struct rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> *__restrict__ local_sort_block_assignments,
		const int *__restrict__ distinction_thresholds,
		const int num_distinction_thresholds)
{
	__shared__ unsigned int subbin_offsets[256];
	unsigned int tloc_count = global_histo[blockIdx.x*256+threadIdx.x];


	CUDA_SHARED_HISTO_EXCL_PFX_SUM_DECLARATIONS;
	CUDA_SHARED_HISTO_EXCL_PFX_SUM(tloc_count, subbin_offsets, true);

	// Every thread of this ext-IN-bin's offset computation now checks if its OUT-bin is: 0, small-locrec, large-locrec or ext-sort
	// LocRec Sorting Assignment
	if(tloc_count>0){
		// LOCREC-SORTING
		if(tloc_count<=distinction_thresholds[num_distinction_thresholds-1]){
			unsigned int idx;

			// TODO improvement: only compute address of atomic_inc ptr in the conditionals and do adds together without branching
			if(tloc_count <= distinction_thresholds[0]){
				idx = atomicAdd(&locrec_per_kpb_assnmt_offsets[0], 1);
			}
			#pragma unroll
			for(int i=1; i < num_distinction_thresholds; i++){
				if(tloc_count > distinction_thresholds[i-1] && tloc_count <= distinction_thresholds[i]){
					idx = atomicAdd(&locrec_per_kpb_assnmt_offsets[i], 1);
				}
			}

			struct rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> tmp;
			// Offset is equal to: offset of the ext-IN-bin PLUS the relative offset of the thread's OUT-bin
			tmp.offset = in_bin_info[blockIdx.x].offset + subbin_offsets[threadIdx.x];
			tmp.num_elements = tloc_count;
			local_sort_block_assignments[idx] = tmp;
		}
		// EXT-SORTING
		else{
			unsigned int idx = atomicAdd(atomic_block_assignment_counter, 1);
			SubBucketInfo tmp;
			tmp.offset = in_bin_info[blockIdx.x].offset + subbin_offsets[threadIdx.x]; 			// STILL TO DO on CPU
			tmp.block_num_elements = tloc_count; 															// STILL TO PROPERLY DO on CPU
			tmp.bin_idx = idx; 																				// TODO figure out if bin index needs to be in order (i.e. leftmost bin has idx 0, ...)

			block_assignments[idx] = tmp;
		}
	}

	global_offsets[in_bin_info[blockIdx.x].bin_idx*256+threadIdx.x] = subbin_offsets[threadIdx.x];
}


template <
	typename KeyT,				// Data type of the keys within device memory. Data will be twiddled-out from unsigned type
	typename ValueT,
	typename IndexT, 			// Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
	int NUM_BITS, 				// Number of bits being sorted at a time
	int KPT,					// Number of keys per thread
	int TPB,					// Number of threads per block
	int IS_LAST_PASS,			//
	int CUB_BITS_PER_PASS = 4	//
>
__global__ void do_locrec_radix_sort_keys(KeyT *__restrict__ keys_in, ValueT *__restrict__ values_in, const struct rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> *__restrict__ block_task_infos, const unsigned char byte, KeyT *__restrict__ keys_out, ValueT *__restrict__ values_out)
{
	/*** TYPEDEFS ***/
	typedef Traits<KeyT>							KeyTraits;
	typedef typename KeyTraits::UnsignedBits	UnsignedBits;
	typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;
	const unsigned int RADIX = 0x01<<NUM_BITS;
	// The offset of the keys of the bucket to which this thread block is assigned to
	const IndexT block_offset = block_task_infos[blockIdx.x].offset;
	const IndexT block_max_keys = block_task_infos[blockIdx.x].offset + block_task_infos[blockIdx.x].num_elements;

	// Specialize BlockRadixSort type for our thread block
	typedef BlockRadixSort<UnsignedBits, TPB, KPT, ValueT, CUB_BITS_PER_PASS, true, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, 1, 1, 520> BlockRadixSortT;

	// Shared memory
	__shared__ union
	LocalSortTempStorage{
		struct {
			union {
				UnsignedBits keys[TPB*KPT];
				ValueT values[cub::If<
				              cub::Equals<ValueT, cub::NullType>::VALUE,
				              cub::Int2Type<1>,
				              cub::Int2Type<TPB*KPT>
							>::Type::VALUE];
			} block_local_sorted;
			unsigned int block_local_histo[RADIX];
			unsigned int block_local_offsets[RADIX];
		} atomic_sort;
		typename BlockRadixSortT::TempStorage   sort;
		LocalSortTempStorage() {}
	} temp_storage;

	// Per-thread tile items
	UnsignedBits tloc_keys[KPT];
	unsigned int tloc_keys_masked[KPT];

	/********************
	 * INIT SHARED HISTO
	 *********************/
	if(TPB == 32 || threadIdx.x < 32){
		#pragma unroll
		for(int i=0;i<RADIX;i+=32){
			temp_storage.atomic_sort.block_local_histo[i+threadIdx.x] = 0;
		}
	}
	__syncthreads();

	/**************************
	 * LOAD KEYS & MASK		  *
	 **************************/
	KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<UnsignedBits, KeyT, 0, KPT>(keys_in, tloc_keys, block_max_keys);
	bool all_kpt_in_bounds = KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_keys, KPT-1);

	// Compute masks
	if(all_kpt_in_bounds){
		#pragma unroll
		for(int i=0; i<KPT; i++){
			tloc_keys_masked[i] = tloc_keys[i]&((0x01<<NUM_BITS)-1);
		}
	}else{
		#pragma unroll
		for(int i=0; i<KPT; i++){
			if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_keys, i)){
				tloc_keys_masked[i] = tloc_keys[i]&((0x01<<NUM_BITS)-1);
			}else{
				tloc_keys_masked[i] = ((0x01<<NUM_BITS)-1);
			}
		}
	}

	/**************************
	 * HISTOGRAM 			  *
	 **************************/
	/*** RLE-HISTO ****/
//	if(all_kpt_in_bounds){
		// Sort (masked) digit values to improve RLE
		SortingNetwork<unsigned int>::sort_runs<(KPT>18?9:((KPT+1)/2))>(tloc_keys_masked);

		unsigned int rle = 1;
		#pragma unroll
		for(int i=1; i<KPT; i++){
			if(tloc_keys_masked[i] == tloc_keys_masked[i-1])
				rle++;
			else{
				atomicAdd(&temp_storage.atomic_sort.block_local_histo[tloc_keys_masked[i-1]], rle);
				rle=1;
			}
			tloc_keys_masked[i-1] = tloc_keys[i-1]&((0x01<<NUM_BITS)-1);
		}
		atomicAdd(&temp_storage.atomic_sort.block_local_histo[tloc_keys_masked[KPT-1]], rle);
		tloc_keys_masked[KPT-1] = tloc_keys[KPT-1]&((0x01<<NUM_BITS)-1);
//	}
	/*** NON-RLE-HISTO ***/
//	else{
//		for(int i=0; i<KPT; i++){
//			if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_keys, i)){
//				atomicAdd(&temp_storage.atomic_sort.block_local_histo[tloc_keys_masked[i]], 1);
//			}
//		}
//	}

	// Make sure we've got the counts from all threads
	__syncthreads();

	/**************************
	 * PREFIX-SUM			  *
	 **************************/
	const int IPT = (TPB-1+RADIX)/TPB;
	const int PFX_SUM_THREADS = ((RDXSRT_WARP_THREADS*IPT-1+RADIX)/(RDXSRT_WARP_THREADS*IPT)) *RDXSRT_WARP_THREADS;
	unsigned int threads_histo_counters[IPT];
	unsigned int max_i;
	unsigned int tmp_max_i;
	#pragma unroll
	for(int i=0;i<IPT;i++){
		if(i + IPT*threadIdx.x < RADIX){
			threads_histo_counters[i] = temp_storage.atomic_sort.block_local_histo[i + IPT*threadIdx.x];
		}else if((PFX_SUM_THREADS*IPT)%RADIX!=0){
			threads_histo_counters[i] = 0;
		}
	}
	ExclusivePrefixSumAndMax<unsigned int, false, IPT, PFX_SUM_THREADS, TPB, RADIX>(threads_histo_counters, temp_storage.atomic_sort.block_local_offsets, max_i, tmp_max_i);

	/**************************
	 * PARTITIONING			  *
	 **************************/
	if(all_kpt_in_bounds){
		#pragma unroll
		for(int i=0; i<KPT; i++){
			tloc_keys_masked[i] = atomicAdd(&temp_storage.atomic_sort.block_local_offsets[tloc_keys_masked[i]], 1);
			temp_storage.atomic_sort.block_local_sorted.keys[tloc_keys_masked[i]] = tloc_keys[i];
		}
	}else{
		for(int i=0; i<KPT; i++){
			if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_keys, i)){
				tloc_keys_masked[i] = atomicAdd(&temp_storage.atomic_sort.block_local_offsets[tloc_keys_masked[i]], 1);
				temp_storage.atomic_sort.block_local_sorted.keys[tloc_keys_masked[i]] = tloc_keys[i];
			}
		}
	}

	// Make sure the result from sorting on the LSD is in shared memory
	__syncthreads();

	/*****************************************************
	 * LAST PASS SORTING (IF BUCKET HASN'T BEEN MERGED)  *
	 *****************************************************/
	/*** IF IT'S ALREADY LAST PASS SORT (AND THE BUCKET IS NOT A MERGED ONE) ***/
	if(IS_LAST_PASS && (RDXSRT_CFG_MERGE_LOCREC_THRESH==0
#if RDXSRT_CFG_MERGE_LOCREC_THRESH
			|| block_task_infos[blockIdx.x].is_merged == 0
#endif
			)){
		/*** WRITE OUT KEYS ***/
		for(int i=threadIdx.x; i<block_task_infos[blockIdx.x].num_elements;i+=TPB){
			UnsignedBits tmp = KeyTraits::TwiddleOut(temp_storage.atomic_sort.block_local_sorted.keys[i]);
			keys_out[block_task_infos[blockIdx.x].offset+i] = (reinterpret_cast<KeyT*>(&tmp))[0];
		}

		/************************
		 * VALUES				*
		 ************************/
		if(cub::Equals<ValueT, cub::NullType>::NEGATE){
			/*** LOAD VALUES ***/
			ValueT values[KPT];
			KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<ValueT, ValueT, 0, KPT>(values_in, values, block_max_keys);

			// Prior to re-using the shared memory (that was used to hold the keys), make sure all threads are done with writing the keys
			__syncthreads();

			/*** SCATTER VALUES TO SHARED MEMORY ***/
			if(all_kpt_in_bounds){
				#pragma unroll
				for(int i=0; i<KPT; i++){
					temp_storage.atomic_sort.block_local_sorted.values[tloc_keys_masked[i]] = values[i];
				}
			}else{
				for(int i=0; i<KPT; i++){
					if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_keys, i)){
						temp_storage.atomic_sort.block_local_sorted.values[tloc_keys_masked[i]] = values[i];
					}
				}
			}
			__syncthreads();

			/*** STORE VALUES ***/
			for(int i=threadIdx.x; i<block_task_infos[blockIdx.x].num_elements;i+=TPB){
				values_out[block_task_infos[blockIdx.x].offset+i] = temp_storage.atomic_sort.block_local_sorted.values[i];
			}
		}
	}

	/**************************************************
	 * SORTING least-significant bits 8 to 32-8*PASS  *
	 **************************************************/
	else{
		unsigned int thread_offset_blocked = KPT * threadIdx.x;
		bool all_kpt_in_bounds_blocked = (thread_offset_blocked+KPT-1<block_task_infos[blockIdx.x].num_elements);
		if(all_kpt_in_bounds_blocked){
			#pragma unroll
			for(int i=0; i<KPT; i++){
				tloc_keys[i] = temp_storage.atomic_sort.block_local_sorted.keys[thread_offset_blocked+i];
			}
		}else{
			for(int i=0; i<KPT; i++){
				if(thread_offset_blocked+i < block_task_infos[blockIdx.x].num_elements){
					tloc_keys[i] = temp_storage.atomic_sort.block_local_sorted.keys[thread_offset_blocked+i];
				}else{
					tloc_keys[i] = UnsignedBits(-1);
				}
			}
		}

		/************************
		 * VALUES				*
		 ************************/
		ValueT values[KPT];
		if(cub::Equals<ValueT, cub::NullType>::NEGATE){
			KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<ValueT, ValueT, 0, KPT>(values_in, values, block_max_keys);

			__syncthreads();
//			if(all_kpt_in_bounds){
//				#pragma unroll
//				for(int i=0; i<KPT; i++){
//					temp_storage.atomic_sort.block_local_sorted.values[tloc_keys_masked[i]] = values[i];
//				}
//			}else{
				for(int i=0; i<KPT; i++){
					if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_keys, i)){
						temp_storage.atomic_sort.block_local_sorted.values[tloc_keys_masked[i]] = values[i];
					}
				}
//			}
			__syncthreads();

//			if(all_kpt_in_bounds_blocked){
				#pragma unroll
				for(int i=0; i<KPT; i++){
					values[i] = temp_storage.atomic_sort.block_local_sorted.values[thread_offset_blocked+i];
				}
//			}else{
//				for(int i=0; i<KPT; i++){
//					if(thread_offset_blocked+i < block_task_infos[blockIdx.x].num_elements){
//						values[i] = temp_storage.atomic_sort.block_local_sorted.values[thread_offset_blocked+i];
//					}
//				}
//			}
		}


		/************************
		 * SUBSEQUENT DIGITS	*
		 ************************/

		// Make sure every thread has its keys after having sorted on the LSD
		__syncthreads();

		// Sort keys
#if RDXSRT_CFG_MERGE_LOCREC_THRESH
		BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(tloc_keys, values, NUM_BITS, (8*sizeof(KeyT))-(NUM_BITS*(byte-block_task_infos[blockIdx.x].is_merged)));
#else
		BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(tloc_keys, values, NUM_BITS, (NUM_BITS*sizeof(KeyT))-(NUM_BITS*byte));
#endif

		// Twiddle out
		#pragma unroll
		for(int i=0; i<KPT; i++){
			tloc_keys[i] = KeyTraits::TwiddleOut(tloc_keys[i]);
		}

		/**************************
		 * STORE SORTED KEYS	  *
		 **************************/
		StoreDirectStriped<TPB, KeyT>(threadIdx.x, keys_out + block_task_infos[blockIdx.x].offset, (reinterpret_cast<KeyT(&)[KPT]>(tloc_keys)), block_task_infos[blockIdx.x].num_elements);
		if(cub::Equals<ValueT, cub::NullType>::NEGATE){
			StoreDirectStriped<TPB, ValueT>(threadIdx.x, values_out + block_task_infos[blockIdx.x].offset, values, block_task_infos[blockIdx.x].num_elements);
		}
	}
}

#endif /* CUDA_RADIX_SORT_H_ */

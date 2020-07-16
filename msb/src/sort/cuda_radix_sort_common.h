#ifndef CUDA_RADIX_SORT_COMMON_H_
#define CUDA_RADIX_SORT_COMMON_H_

#include "sort/cuda_radix_sort_config.h"

/*******************************************
 * LOADING FROM GLOBAL MEMORY TO REGISTERS *
 *******************************************/

enum RDXSRT_LOAD_STRIDE {
	RDXSRT_LOAD_STRIDE_SINGLE = 1,
	RDXSRT_LOAD_STRIDE_WARP = RDXSRT_WARP_THREADS,
	RDXSRT_LOAD_STRIDE_BLOCK = 33,
};

template <
	typename IndexT,
	enum RDXSRT_LOAD_STRIDE LOAD_STRIDE,
	int KPT,
	int TPB,
	unsigned int CFG_STRIDE = LOAD_STRIDE
>
struct LoadUnit
{
	const IndexT thread_base_offset;

	__device__ __forceinline__ LoadUnit(IndexT block_base_offset, IndexT thread_id)
		: thread_base_offset( block_base_offset + ((thread_id / CFG_STRIDE) * (CFG_STRIDE*KPT)) + (thread_id%CFG_STRIDE))
	{}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given iterator offset)
	>
	__device__ __forceinline__ void LoadStrided(const LoadT *in_data, RegisterT (&t_data)[KPT])
	{
		#pragma unroll
		for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
			t_data[ITERATOR_OFFSET+i] = (reinterpret_cast<const RegisterT*>(in_data))[thread_base_offset + (i+ITERATOR_OFFSET) * CFG_STRIDE];
		}
	}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given iterator offset)
	>
	__device__ __forceinline__ void LoadStridedRun(const LoadT *in_data, RegisterT (&t_data)[NUM_ITEMS])
	{
		#pragma unroll
		for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
			t_data[i] = (reinterpret_cast<RegisterT*>(in_data))[thread_base_offset + (i+ITERATOR_OFFSET) * CFG_STRIDE];
		}
	}

	__device__ __forceinline__ bool ThreadRequiresGuards(IndexT base_offset_plus_block_elements)
	{
		return !ThreadIndexInBounds(base_offset_plus_block_elements, KPT-1);
	}

	__device__ __forceinline__ bool ThreadIndexInBounds(IndexT base_offset_plus_block_elements, int index)
	{
		return (thread_base_offset + index * CFG_STRIDE < base_offset_plus_block_elements);
	}

	__device__ __forceinline__ IndexT GetAbsIndexForThreadIndex(int index)
	{
		return thread_base_offset + index * CFG_STRIDE;
	}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given offset)
	>
	__device__ __forceinline__ void LoadStridedWithGuards(const LoadT *in_data, RegisterT (&t_data)[KPT], IndexT base_offset_plus_block_elements)
	{
		// All of thread's items in bounds anyway
		if(ThreadIndexInBounds(base_offset_plus_block_elements, ITERATOR_OFFSET+NUM_ITEMS-1)){
			LoadStrided<RegisterT, LoadT, ITERATOR_OFFSET, NUM_ITEMS>(in_data, t_data);
		}
		// Some of thread's items out of bounds
		else{
			#pragma unroll
			for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
				if(ThreadIndexInBounds(base_offset_plus_block_elements, i+ITERATOR_OFFSET))
					t_data[ITERATOR_OFFSET+i] = (reinterpret_cast<const RegisterT*>(in_data))[thread_base_offset + (i+ITERATOR_OFFSET) * CFG_STRIDE];
			}
		}
	}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given offset)
	>
	__device__ __forceinline__ void WriteBackStridedWithGuards(const RegisterT (&t_data)[KPT], LoadT *out_data, IndexT base_offset_plus_block_elements)
	{
		// All of thread's items in bounds anyway
		if(false && ThreadIndexInBounds(base_offset_plus_block_elements, ITERATOR_OFFSET+NUM_ITEMS-1)){
			// TODO implement
//			WriteBackStrided<RegisterT, LoadT, ITERATOR_OFFSET, NUM_ITEMS>(out_data, t_data);
		}
		// Some of thread's items out of bounds
		else{
			#pragma unroll
			for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
				if(ThreadIndexInBounds(base_offset_plus_block_elements, i+ITERATOR_OFFSET))
					out_data[thread_base_offset + (i+ITERATOR_OFFSET) * CFG_STRIDE] = reinterpret_cast<const LoadT*>(t_data)[ITERATOR_OFFSET+i];
			}
		}
	}
};

template <
	typename 	IndexT,						// Type used for computing offsets (e.g. uint // unsigned long long int)
    int			KPT,						// Number of keys per thread
    int 		TPB							// Number of threads per block
>
struct LoadUnit <IndexT, RDXSRT_LOAD_STRIDE_SINGLE, KPT, TPB, RDXSRT_LOAD_STRIDE_SINGLE>
{
	const IndexT thread_base_offset;

	__device__ __forceinline__ LoadUnit(IndexT block_base_offset, IndexT thread_id)
		: thread_base_offset( block_base_offset + (thread_id*KPT))
	{}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given iterator offset)
	>
	__device__ __forceinline__ void LoadStrided(const LoadT *in_data, RegisterT (&t_data)[KPT])
	{
		#pragma unroll
		for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
			t_data[ITERATOR_OFFSET+i] = (reinterpret_cast<const RegisterT*>(in_data))[thread_base_offset + (i+ITERATOR_OFFSET)];
		}
	}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given iterator offset)
	>
	__device__ __forceinline__ void LoadStridedRun(const LoadT *in_data, RegisterT (&t_data)[NUM_ITEMS])
	{
		#pragma unroll
		for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
			t_data[i] = (reinterpret_cast<const RegisterT*>(in_data))[thread_base_offset + (i+ITERATOR_OFFSET)];
		}
	}

	__device__ __forceinline__ bool ThreadRequiresGuards(IndexT base_offset_plus_block_elements)
	{
		return (!ThreadIndexInBounds(base_offset_plus_block_elements, KPT-1));
	}

	__device__ __forceinline__ bool ThreadIndexInBounds(IndexT base_offset_plus_block_elements, int index)
	{
		return (thread_base_offset + index < base_offset_plus_block_elements);
	}

	__device__ __forceinline__ IndexT GetAbsIndexForThreadIndex(int index)
	{
		return thread_base_offset + index;
	}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given offset)
	>
	__device__ __forceinline__ void LoadStridedWithGuards(const LoadT *in_data, RegisterT (&t_data)[KPT], IndexT base_offset_plus_block_elements)
	{
		// All of thread's items in bounds anyway
		if(ThreadIndexInBounds(base_offset_plus_block_elements, ITERATOR_OFFSET+NUM_ITEMS-1)){
			LoadStrided<RegisterT, LoadT, ITERATOR_OFFSET, NUM_ITEMS>(in_data, t_data);
		}
		// Some of thread's items out of bounds
		else{
			#pragma unroll
			for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
				if(ThreadIndexInBounds(base_offset_plus_block_elements, i+ITERATOR_OFFSET))
					t_data[ITERATOR_OFFSET+i] = (reinterpret_cast<const RegisterT*>(in_data))[thread_base_offset + (i+ITERATOR_OFFSET)];
			}
		}
	}

	template<
		typename 	RegisterT, 					// Target re-interpretation cast type (type of thread's registers)
		typename 	LoadT, 						// Source re-interpretation cast type (type of data)
		int 		ITERATOR_OFFSET, 			// Iterator offset (starting to fetch the (i+1)-th key of the thread
		int 		NUM_ITEMS 					// Iteration count (number of items to fetch, starting from the given offset)
	>
	__device__ __forceinline__ void WriteBackStridedWithGuards(const RegisterT (&t_data)[KPT], LoadT *out_data, IndexT base_offset_plus_block_elements)
	{
		// All of thread's items in bounds anyway
		if(false && ThreadIndexInBounds(base_offset_plus_block_elements, ITERATOR_OFFSET+NUM_ITEMS-1)){
			// TODO implement
//			LoadStrided<RegisterT, LoadT, ITERATOR_OFFSET, NUM_ITEMS>(out_data, t_data);
		}
		// Some of thread's items out of bounds
		else{
			#pragma unroll
			for(int i=0; i<(NUM_ITEMS<(KPT-ITERATOR_OFFSET)?NUM_ITEMS:(KPT-ITERATOR_OFFSET)); i++){
				if(ThreadIndexInBounds(base_offset_plus_block_elements, i+ITERATOR_OFFSET))
					out_data[thread_base_offset + (i+ITERATOR_OFFSET)] = reinterpret_cast<const LoadT*>(t_data)[ITERATOR_OFFSET+i];
			}
		}
	}
};


/***************
 * PREFIX-SUMS *
 ***************/
/**
 * Number of threads (TPB) contributing to the prefix sum must be an integer multiple of the warp size (e.g. 32). Any thread that doesn't contribute but is included in the number of threads must initialize its non-contributing items to 0.
 * For instance, 30 items (NUM_ITEMS), 2 items per thread (IPT) with warp size of 32 would require 30/2=15 contributing threads (TPB). Yet, the threads must be padded to a full warp, hence TPB=32. With the latter 17 threads not contributing any item their items must be 0.
 * @param items
 * @param pfx_sum_out
 * @param max
 * @param max_idx
 */
template<
typename 	ItemT,		// Numberical data type to be summed
bool 		COMPUTE_MAX,// Whether to also scan the maximum item (if so, max will hold the maximum, max_idx the zero based index)
int 		IPT,		// Number of items per thread
int 		TPB,		// Number of threads involved in the prefix-sum
int 		KERNEL_TPB,	// Number of threads of the thread block
int 		NUM_ITEMS	// Number of items to compute the prefix-sum over: NUM_ITEMS <= TPB*IPT
>
__device__ __forceinline__ void ExclusivePrefixSumAndMax(unsigned int (&items)[IPT], unsigned int *pfx_sum_out, unsigned int &max, unsigned int &max_idx)
{
	// For each warp writing it's sum there
	volatile __shared__ unsigned int cache[(RDXSRT_WARP_THREADS-1+TPB)/RDXSRT_WARP_THREADS];
	volatile __shared__ unsigned int max_cache[(RDXSRT_WARP_THREADS-1+TPB)/RDXSRT_WARP_THREADS];

	const unsigned int lane_id = threadIdx.x % RDXSRT_WARP_THREADS;
	const unsigned int warp_id = threadIdx.x / RDXSRT_WARP_THREADS;

	/*** PER-THREAD-REDUCTION ***/
	unsigned int sum = 0;
	unsigned int tmp;
	if(COMPUTE_MAX){
		max = 0;//std::numeric_limits<unsigned int>::min();
		max_idx = 0;
	}

	if(TPB == KERNEL_TPB || threadIdx.x<TPB){
		// Find thread's maximum
		#pragma unroll
		for(int i=0; i<IPT; i++){
			sum += items[i];
			items[i] = sum;
			if(COMPUTE_MAX){
				if(items[i]>max){
					max = items[i];
					max_idx = threadIdx.x * IPT+i;
				}
			}
		}

		/*** PACK MAX VALUE WITH MAX INDEX: [ MAX | MAX | MAX_IDX | MAX_IDX ] ***/
		if(COMPUTE_MAX)
			max = (max << 16) + max_idx;

		/*** WARP-SYNC ***/
		#pragma unroll
		for(int i=1; i<RDXSRT_WARP_THREADS;i*=2){
			tmp = __shfl_up(sum, i);
			if(lane_id >=  i)
				sum += tmp;
			if(COMPUTE_MAX){
				max_idx = __shfl_up(max, i);
				max = max_idx > max ? max_idx : max;
			}
		}

		tmp = sum-items[IPT-1];
		#pragma unroll
		for(int i=0; i<IPT; i++){
			items[i] += tmp;
		}

		// The last lane of each warp now holds the inclusive prefix-sum of the warp
		if(lane_id == RDXSRT_WARP_THREADS-1){
			cache[warp_id] = sum;
			if(COMPUTE_MAX)
				max_cache[warp_id] = max;
		}
	}

	/*** Make sure each warp has written its prefix-sum ***/
	__syncthreads();

	// Every thread is assigned to one PER-WARP SUM
	if(warp_id == 0 && lane_id<(RDXSRT_WARP_THREADS-1+TPB)/RDXSRT_WARP_THREADS){
		#pragma unroll
		for(int i=1;i<(RDXSRT_WARP_THREADS-1+TPB)/RDXSRT_WARP_THREADS;i*=2){
			if(lane_id >= i){
				cache[lane_id] += cache[lane_id-i];
				if(COMPUTE_MAX){
					if(max_cache[lane_id-i]>max_cache[lane_id])
						max_cache[lane_id] = max_cache[lane_id-i];
				}
			}
		}
	}
	/* Make sure the per warp sums are available */
	__syncthreads();

	if(COMPUTE_MAX){
		max_idx = max_cache[((RDXSRT_WARP_THREADS-1+TPB)/RDXSRT_WARP_THREADS) - 1];
		max = max_idx >> 16;
		max_idx = max_idx & 0x0000FFFF;
	}

	if(TPB == KERNEL_TPB || threadIdx.x<TPB){
		/*** INCLUDE THE PER-WARP OFFSET FOR WARPS >0 ***/
		if(warp_id > 0){
			#pragma unroll
			for(int i=0; i<IPT; i++){
				items[i] += cache[warp_id-1];
			}
		}

		/*** WRITE GLOBAL OFFSETS TO SHARED MEMORY ***/
		#pragma unroll
		for(int i=0; i<IPT; i++){
			if(threadIdx.x*IPT+i<NUM_ITEMS-1)
				pfx_sum_out[threadIdx.x*IPT+i+1] = items[i];
			else if(threadIdx.x*IPT+i==NUM_ITEMS-1)
				pfx_sum_out[0] = 0;
		}
	}
	__syncthreads();
}


// For external sorting
//template<
//	typename IndexT
//>
struct SubBucketInfo {
	unsigned int offset;				// Offset for the *BLOCK* within keys[] (used to calculate the block's in-keys)
	int block_num_elements;				// Number of elements the block should work on, typically equals KPB
	unsigned int bin_idx;				// The bin's index, used to determine the target address of histogram etc.
};

// For external sorting
struct rdxsrt_extsrt_block_to_bin_t {
	unsigned int offset;				// Offset for the *BLOCK* within keys[] (used to calculate the block's in-keys)
	int block_num_elements;				// Number of elements the block should work on, typically equals KPB
	unsigned int bin_idx;				// The bin's index, used to determine the target address of histogram etc.
	unsigned int bin_start_offset;		// Offset indicating where within keys[] the *BIN* for this block starts (used to write out results)
};

// For local recursive sorting
template<
	int TINY_BUCKET_MERGE_THRESHOLD
>
struct rdxsrt_recsrt_block_info_t {
	unsigned int offset;				// The bucket's offset within the input
	unsigned int num_elements;			// Number of elements within this sub-bucket
	short is_merged;
	__device__ __inline__ void set_merged(bool to_merged){ is_merged = to_merged; };
	__device__ __inline__ bool get_merged(){ return is_merged; };
};

template<>
struct rdxsrt_recsrt_block_info_t<0> {
	unsigned int offset;				// The bucket's offset within the input
	unsigned int num_elements;			// Number of elements within this sub-bucket
	__device__ __inline__ void set_merged(bool to_merged){ };
	__device__ __inline__ bool get_merged(){ return false; };
};


/***************************
 *** STRIDED LOAD MACROS ***
 ***************************/
#define RDXSRT_STRIDE_SINGLE 0
#define RDXSRT_STRIDE_WARP 1
#define RDXSRT_CFG_STRIDE RDXSRT_STRIDE_WARP

#if RDXSRT_CFG_STRIDE == RDXSRT_STRIDE_SINGLE
#define RDXSRT_STRIDED_GET_THREAD_OFFSET(thread_index, keys_per_thread) (thread_index * keys_per_thread)
#define RDXSRT_STRIDED_GET_KEY_IDX(block_offset, thread_offset, loop_iteration) (block_offset + thread_offset+loop_iteration)
#define RDXSRT_STRIDED_ALL_KEYS_INBOUNDS_COND(block_num_elements, thread_offset, keys_per_thread) (thread_offset+keys_per_thread-1<block_num_elements)
#define RDXSRT_STRIDED_MAX_ITERATION_COND(block_num_elements, thread_offset, keys_per_thread, loop_iterator) (thread_offset+loop_iterator < block_num_elements)
#elif RDXSRT_CFG_STRIDE == RDXSRT_STRIDE_WARP
#define RDXSRT_STRIDED_GET_THREAD_OFFSET(thread_index, keys_per_thread) (((thread_index / 32)*32*keys_per_thread)+(thread_index%32))
#define RDXSRT_STRIDED_GET_KEY_IDX(block_offset, thread_offset, loop_iteration) (block_offset + thread_offset + loop_iteration*32)
#define RDXSRT_STRIDED_ALL_KEYS_INBOUNDS_COND(block_num_elements, thread_offset, keys_per_thread) (thread_offset+(keys_per_thread-1)*32<block_num_elements)
//#define RDXSRT_STRIDED_ALL_KEYS_INBOUNDS_COND(block_num_elements, thread_offset, keys_per_thread) (((threadIdx.x / 32 + 1)*32*keys_per_thread)-1<block_num_elements) /* WARP GRANULARITY CONDITION */
#define RDXSRT_STRIDED_MAX_ITERATION_COND(block_num_elements, thread_offset, keys_per_thread, loop_iterator) (thread_offset+loop_iterator*32<((block_num_elements)))
#endif

// HELPER MACROS
#define RDXSRT_WPB(num_threads) (1+(num_threads-1)/32)
#define RDXSRT_MAX(a,b) ((a)>(b)?(a):(b))
#define RDXSRT_NUM_BITS(n) (n<=32?5:(n<=64?6:(n<=128?7:(n<=256?8:(n<=512?9:10)))))



/**
 * Declarations required for CUDA_SHARED_HISTO_EXCL_PFX_SUM().
 */
#define CUDA_SHARED_HISTO_EXCL_PFX_SUM_DECLARATIONS	\
	const int warp_id = threadIdx.x / 32;						\
	const int lane_id = threadIdx.x % 32;						\
																\
	/* Shared memory for per warp sum */						\
	volatile __shared__ unsigned int cache[32];					\
	unsigned int sum[1];										\
	unsigned int tmp;

#define CUDA_SHARED_MAX_DECLARATIONS	\
	unsigned int max_i;					\
	unsigned int tmp_max_i;				\
	volatile __shared__ unsigned int max_cache[32];

#define CUDA_SHARED_HISTO_EXCL_PFX_SUM_LESS_TPB_DECLARATIONS(TPB)	\
	const int warp_id = threadIdx.x / 32;						\
	const int lane_id = threadIdx.x % 32;						\
																\
	/* Shared memory for per warp sum */						\
	volatile __shared__ unsigned int cache[8];					\
	unsigned int sum[(TPB>256)?1:(256/TPB)];					\
	unsigned int tmp;


/**
 * Computes the exclusive prefix sum over 256 elements (taken from count_array) and writes the result to offset_array. count_array may be the same as offset_array.
 * First, each of the 256 threads fetches exactly one element. Then a Kagge-Stone prefix sum is computed for each warp individually.
 * Secondly, a prefix sum over the warps is computed.
 * Finally, the warp prefix sums are down-swept to the threads again.
 *
 * REQUIRES AT LEAST 256 Threads per block.
 */
#define CUDA_SHARED_HISTO_EXCL_PFX_SUM(count_array, offset_array, exclusive)	\
																\
	/* Each thread starts out with one of the 256 elements*/	\
	sum[0] = 0;													\
																\
	/* Kogge-Stone (for per warp partial prefix-sum)*/			\
	if(threadIdx.x < 256){										\
		sum[0] = (count_array);									\
		tmp = __shfl_up(sum[0], 1);								\
		if(lane_id >=  1)										\
			sum[0] += tmp;										\
		tmp = __shfl_up(sum[0], 2);								\
		if(lane_id >=  2)										\
			sum[0] += tmp;										\
		tmp = __shfl_up(sum[0], 4);								\
		if(lane_id >=  4)										\
			sum[0] += tmp;										\
		tmp = __shfl_up(sum[0], 8);								\
		if(lane_id >=  8)										\
			sum[0] += tmp;										\
		tmp = __shfl_up(sum[0], 16);							\
		if(lane_id >=  16)										\
			sum[0] += tmp;										\
																\
		/* Write out per warp sum*/								\
		if(lane_id == 31){										\
			cache[warp_id] = sum[0];							\
		}														\
	}															\
	/* Make sure all warp have written their sum */				\
	__syncthreads();											\
																\
	/* Compute prefix sums over the individual warps*/			\
	if(warp_id == 0 && lane_id < 8){							\
		if(lane_id >= 1)										\
			cache[lane_id] += cache[lane_id-1];					\
		if(lane_id >= 2)										\
			cache[lane_id] += cache[lane_id-2];					\
		if(lane_id >= 4)										\
			cache[lane_id] += cache[lane_id-4];					\
	}															\
	/* Make sure the per warp sums are available */				\
	__syncthreads();											\
																\
	/* Include the warp's offset, for warp 1..7 */				\
	if(warp_id > 0)												\
		sum[0] += cache[warp_id-1];								\
																\
	/* Write to global offset*/									\
	if(exclusive){												\
		if(threadIdx.x<255)										\
			offset_array[threadIdx.x+1] = sum[0];				\
		else 													\
			offset_array[0] = 0;								\
	}else{														\
		if(threadIdx.x<256)										\
			offset_array[threadIdx.x] = sum[0];					\
	}															\
	__syncthreads();

#define CUDA_GENERAL_SHARED_HISTO_EXCL_PFX_SUM(count_array, offset_array, exclusive)	\
																\
	/* Each thread starts out with one of the 256 elements*/	\
	sum[0] = 0;													\
																\
	/* Kogge-Stone (for per warp partial prefix-sum)*/			\
	sum[0] = (count_array);									\
	tmp = __shfl_up(sum[0], 1);								\
	if(lane_id >=  1)										\
		sum[0] += tmp;										\
	tmp = __shfl_up(sum[0], 2);								\
	if(lane_id >=  2)										\
		sum[0] += tmp;										\
	tmp = __shfl_up(sum[0], 4);								\
	if(lane_id >=  4)										\
		sum[0] += tmp;										\
	tmp = __shfl_up(sum[0], 8);								\
	if(lane_id >=  8)										\
		sum[0] += tmp;										\
	tmp = __shfl_up(sum[0], 16);							\
	if(lane_id >=  16)										\
		sum[0] += tmp;										\
															\
	/* Write out per warp sum*/								\
	if(lane_id == 31){										\
		cache[warp_id] = sum[0];							\
	}														\
	/* Make sure all warp have written their sum */				\
	__syncthreads();											\
																\
	/* Compute prefix sums over the individual warps*/			\
	if(warp_id == 0 && lane_id < 16){							\
		if(lane_id >= 1)										\
			cache[lane_id] += cache[lane_id-1];					\
		if(lane_id >= 2)										\
			cache[lane_id] += cache[lane_id-2];					\
		if(lane_id >= 4)										\
			cache[lane_id] += cache[lane_id-4];					\
		if(lane_id >= 8)										\
			cache[lane_id] += cache[lane_id-8];					\
	}															\
	/* Make sure the per warp sums are available */				\
	__syncthreads();											\
																\
	/* Include the warp's offset, for warp 1..7 */				\
	if(warp_id > 0)												\
		sum[0] += cache[warp_id-1];								\
																\
	/* Write to global offset*/									\
	if(exclusive){												\
		offset_array[threadIdx.x+1] = sum[0];					\
		if(threadIdx.x==0)										\
			offset_array[0] = 0;								\
	}else{														\
		offset_array[threadIdx.x] = sum[0];						\
	}															\
	__syncthreads();


// Number of sums per thread, given that we only have TPB threads per block
#define NUM_SPT(TPB) ((TPB>256)?1:(256/TPB))

#define CUDA_SHARED_HISTO_EXCL_PFX_SUM_DYNAMIC(count_array, offset_array, tpb)	\
																\
	/* Each thread starts out with one of the 256 elements*/	\
	/*  0          1     2     3 */								\
	/*x 0 1 2 | 3 4 5 6*/										\
	/* Kogge-Stone (for per warp partial prefix-sum)*/			\
	if(tpb < 256 || threadIdx.x < 256){							\
		if(threadIdx.x == 0){									\
			sum[0] = 0;											\
			for(int x=1;x<NUM_SPT(tpb);x++)						\
				sum[x] = count_array[x-1];						\
		}else{													\
			for(int x=0;x<NUM_SPT(tpb);x++)						\
				sum[x] = count_array[threadIdx.x*NUM_SPT(tpb)-1+x];		\
		}														\
		for(int x=1;x<NUM_SPT(tpb);x++)							\
			sum[x] += sum[x-1];									\
		tmp = __shfl_up(sum[NUM_SPT(tpb)-1], 1);				\
		if(lane_id >=  1){										\
			for(int x=0;x<NUM_SPT(tpb);x++)						\
				sum[x] += tmp;									\
		}														\
		tmp = __shfl_up(sum[NUM_SPT(tpb)-1], 2);				\
		if(lane_id >=  2){										\
			for(int x=0;x<NUM_SPT(tpb);x++)						\
				sum[x] += tmp;									\
		}														\
		tmp = __shfl_up(sum[NUM_SPT(tpb)-1], 4);				\
		if(lane_id >=  4){										\
			for(int x=0;x<NUM_SPT(tpb);x++)						\
				sum[x] += tmp;									\
		}														\
		tmp = __shfl_up(sum[NUM_SPT(tpb)-1], 8);				\
		if(lane_id >=  8){										\
			for(int x=0;x<NUM_SPT(tpb);x++)						\
				sum[x] += tmp;									\
		}														\
		tmp = __shfl_up(sum[NUM_SPT(tpb)-1], 16);				\
		if(lane_id >=  16){										\
			for(int x=0;x<NUM_SPT(tpb);x++)						\
				sum[x] += tmp;									\
		}														\
																\
		/* Write out per warp sum*/								\
		if(lane_id == 31){										\
			cache[warp_id] = sum[NUM_SPT(tpb)-1];				\
		}														\
	}															\
	/* Make sure all warp have written their sum */				\
	__syncthreads();											\
																\
	/* Compute prefix sums over the individual warps*/			\
	if(warp_id == 0 && lane_id < 8){							\
		if(tpb>32 && lane_id >= 1)								\
			cache[lane_id] += cache[lane_id-1];					\
		if(tpb>64 && lane_id >= 2)								\
			cache[lane_id] += cache[lane_id-2];					\
		if(tpb>128 && lane_id >= 4)								\
			cache[lane_id] += cache[lane_id-4];					\
	}															\
	/* Make sure the per warp sums are available */				\
	__syncthreads();											\
																\
	/* Include the warp's offset, for warp 1..7 */				\
	if(warp_id > 0 && tpb > 32){								\
		for(int x=0;x<NUM_SPT(tpb);x++)							\
			sum[x] += cache[warp_id-1];							\
	}															\
																\
	/* Write to global offset*/									\
	if(tpb < 256 || threadIdx.x < 256){							\
		for(int x=0;x<NUM_SPT(tpb);x++)							\
			offset_array[threadIdx.x*NUM_SPT(tpb)+x] 	= sum[x];\
	}															\
	__syncthreads();



#endif /* CUDA_RADIX_SORT_COMMON_H_ */

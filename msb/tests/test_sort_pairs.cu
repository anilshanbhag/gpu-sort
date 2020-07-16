#include <algorithm>
#include "gtest/gtest.h"
#include "cub/cub.cuh"
#include "benchmark/benchmark.h"
#include "sort/gpu_radix_sort.h"
#include "gpu_helper/gpu_warmup.cuh"
#include "data_gen.h"
#include "cli_args.h"

static CachingDeviceAllocator  g_allocator(true);

template<
	typename KeyT,
	typename ValueT
>
void cub_sort_pairs(const KeyT *h_keys, const ValueT *h_values, const unsigned int num_items, KeyT *h_keys_sorted, ValueT *h_values_sorted, const bool secondary_sort_values = true)
{
	// Allocate device memory for input/output
    DoubleBuffer<KeyT> d_keys;
    DoubleBuffer<ValueT> d_values;
    CubDebugExit( g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(KeyT) * num_items) );
    CubDebugExit( g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items) );
    CubDebugExit( g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(ValueT) * num_items) );
    CubDebugExit( g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(ValueT) * num_items) );

    // Initialize device arrays
    CubDebugExit( cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice) );
    CubDebugExit( cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice) );

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items) );
    CubDebugExit( g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes) );

    // Sort
    if(secondary_sort_values)
    	CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_values, d_keys, num_items) );
    BM_START_CUDA_EVENT(cub_sort, cub_sort, NULL);
    CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items) );
    BM_STOP_CUDA_EVENT(cub_sort, cub_sort, NULL);

    // Copy results for verification. GPU-side part is done.
    CubDebugExit( cudaMemcpy(h_keys_sorted, d_keys.Current(), sizeof(KeyT) * num_items, cudaMemcpyDeviceToHost) );
    CubDebugExit( cudaMemcpy(h_values_sorted, d_values.Current(), sizeof(ValueT) * num_items, cudaMemcpyDeviceToHost) );

    // Cleanup
    if (d_keys.d_buffers[0])
    	CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
    if (d_keys.d_buffers[1])
    	CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
    if (d_values.d_buffers[0])
    	CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
    if (d_values.d_buffers[1])
    	CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
    if (d_temp_storage)
    	g_allocator.DeviceFree(d_temp_storage);
}

template<
	typename KeyT,
	typename ValueT
>
::testing::AssertionResult verify_sorted_pairs(KeyT *h_keys, ValueT *h_values, const KeyT *h_keys_sorted, ValueT *h_values_sorted, const unsigned int num_pairs, const bool verify_values)
{
	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;

	// Sort key-value pairs for verification
	cub_sort_pairs<KeyT, ValueT>(h_keys, h_values, num_pairs, h_keys, h_values, verify_values);

	// Compare keys for verification
	int cmp = memcmp(h_keys, h_keys_sorted, num_pairs * sizeof(KeyT));

	// Results are identical
	if(cmp != 0){
		return ::testing::AssertionFailure() << "Keys have not been sorted properly";
	}

	// As our radix sort is unstable, we do a secondary sort on values for easy verification
	if(verify_values){
		if(num_pairs>0){
			unsigned int start_idx = 0;
			KeyT current_key = h_keys_sorted[0];
			for(int i = 1; i < num_pairs; i++){
				// If this key is different from the previous one
				if(current_key != h_keys_sorted[i]){
					if(i-start_idx > 1){
						std::sort(&h_values_sorted[start_idx], &h_values_sorted[i]);
					}
					current_key = h_keys_sorted[i];
					start_idx = i;
				}
				// If it's the last key and current_key has the same value
				else if(i == num_pairs-1 && current_key == h_keys_sorted[i]){
					if(num_pairs-start_idx > 1){
						std::sort(&h_values_sorted[start_idx], &h_values_sorted[num_pairs]);
					}
				}
			}
		}

		// Compare values for verification
		cmp = memcmp(h_values, h_values_sorted, num_pairs * sizeof(ValueT));

		// Values are identical
		if(cmp != 0){
			return ::testing::AssertionFailure();
		}
	}

	return ::testing::AssertionSuccess();
}

template <
	typename KeyT,
	typename ValueT
>
void test_sort_pairs(unsigned int num_pairs, int entropy_level, unsigned long long int seed, bool fast_value_check)
{
	typedef Traits<KeyT>                        	KeyTraits;
	typedef Traits<ValueT>                        	ValueTraits;
	typedef typename KeyTraits::UnsignedBits    	UnsignedKeyBits;
	typedef typename ValueTraits::UnsignedBits    	UnsignedValueBits;

	// Allocate memory for data
	KeyT *h_keys_in = (KeyT *)malloc(num_pairs*sizeof(*h_keys_in));
	KeyT *h_keys_cpy = (KeyT *)malloc(num_pairs*sizeof(*h_keys_cpy));
	KeyT *h_keys_sorted = (KeyT *)malloc(num_pairs*sizeof(*h_keys_sorted));
	ValueT *h_values_in = (ValueT *)malloc(num_pairs*sizeof(*h_values_in));
	ValueT *h_values_cpy = (ValueT *)malloc(num_pairs*sizeof(*h_values_cpy));
	ValueT *h_values_sorted = (ValueT *)malloc(num_pairs*sizeof(*h_values_sorted));

	// For fast value checking
	KeyT *value_to_key_map;

	// Prepare data
	generate_random_keys<UnsignedKeyBits>(reinterpret_cast<UnsignedKeyBits*>(h_keys_in), num_pairs, 0, entropy_level);

	// Prepare values
	if(fast_value_check){
		generate_enumerated_values<UnsignedValueBits>(reinterpret_cast<UnsignedValueBits *>(h_values_in), num_pairs);
		value_to_key_map = (KeyT*)malloc(num_pairs*sizeof(*h_keys_in));
		for(int i=0; i < num_pairs; i++){
			value_to_key_map[reinterpret_cast<UnsignedValueBits *>(h_values_in)[i]] = h_keys_in[i];
		}
	}else{
		generate_random_keys<ValueT>(h_values_in, num_pairs, 0);
	}

	// Copy data for verification
	memcpy(h_keys_cpy, h_keys_in, num_pairs*sizeof(*h_keys_in));
	memcpy(h_values_cpy, h_values_in, num_pairs*sizeof(*h_values_in));

	// Sort
	rdxsrt_warmup_gpu();
	BM_START_CUDA_EVENT(cuda time, sort_total, NULL);
	rdxsrt_unstable_sort_pairs<KeyT, ValueT>(h_keys_in, h_values_in, num_pairs, h_keys_sorted, h_values_sorted);
	BM_STOP_CUDA_EVENT(cuda time, sort_total, NULL);

	// Verification
	BM_START_CPU_PROFILE(cpu verification, cpu_verif);
	ASSERT_TRUE( (verify_sorted_pairs<KeyT, ValueT>(h_keys_cpy, h_values_cpy, h_keys_sorted, h_values_sorted, num_pairs, !fast_value_check)) ) << "Test: \n" << " - key count: " << num_pairs << "\n - entropy level: " << entropy_level << "\n";;

	// Verify values
	if(fast_value_check){
		bool all_values_equal = true;
		size_t total_sum = 0;
		for(int i=0; i < num_pairs; i++){
			total_sum += h_values_sorted[i];
			all_values_equal = all_values_equal && h_values_sorted[i] < num_pairs && (value_to_key_map[reinterpret_cast<UnsignedValueBits *>(h_values_sorted)[i]] == h_keys_sorted[i]);
		}
		all_values_equal = all_values_equal && (total_sum == (size_t)num_pairs * ((size_t)num_pairs-1) / 2);
		ASSERT_TRUE(all_values_equal);
		free(value_to_key_map);
	}
	BM_STOP_CPU_PROFILE(cpu verification, cpu_verif);

	// Free resources
	free(h_keys_in);
	free(h_keys_cpy);
	free(h_keys_sorted);
	free(h_values_in);
	free(h_values_cpy);
	free(h_values_sorted);
}

template <
	typename KeyT,
	typename ValueT
>
void run_test_over_entropies(unsigned int num_pairs, unsigned int repeat_count = 3, int *entropy_levels = NULL, int num_entropies = 0, const char *profile_name = "sort_pairs")
{
	static int default_entropy_levels[] = {1,2,3,4,5,6,7,8,9,10,11,0};
	if(!entropy_levels){
		entropy_levels = default_entropy_levels;
		num_entropies = sizeof(default_entropy_levels)/sizeof(default_entropy_levels[0]);
	}

	for(int j=0; j<num_entropies; j++){
		int entropy_level = entropy_levels[j];
		double bit_probability = pow(0.5, (double)entropy_level);
		double single_bit_entropy = entropy_level<=0 ? 0.0 : ((-bit_probability)*log2(bit_probability)+(-(1-bit_probability))*log2(1-bit_probability));
		double entropy = sizeof(KeyT)*8 * single_bit_entropy;

		for(int i=0; i<repeat_count; i++){
			printf(" --- SORTPAIRS.ENTROPIES (Entropy Lev. (bit entropy): %2d (%6.3f), Iteration: %2d - key_count: %u)---\n", entropy_level, entropy, i, num_pairs);

			BM_OPEN_PROFILE(profile_name);
			BM_SET_METRIC_UINT(num_keys, num_keys, num_pairs);
			BM_SET_METRIC_INT(entr_and, entr_and, entropy_level);
			BM_SET_METRIC_DOUBLE(bit_entr, bit_entropy, (entropy_level<=0?0.0:entropy));
			test_sort_pairs<KeyT, ValueT>(num_pairs, entropy_level, i, true);
			BM_CLOSE_PROFILE();
		}
	}
}


/*****************************
 *** CONSTANT PROBLEM SIZE ***
 *****************************/
TEST (Sort_Pairs, Entropy_UINT_UINT) {
	typedef unsigned int KeyT;
	typedef unsigned int ValueT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/(sizeof(KeyT)+sizeof(ValueT))) : sort_pairs_default_prob_size;

	run_test_over_entropies<KeyT, ValueT>(num_keys, sort_repeat_count, NULL, 0, "sort_pairs_UINT_UINT");
}

TEST (Sort_Pairs, Entropy_UINT_UINT64) {
	typedef unsigned int KeyT;
	typedef unsigned long long int ValueT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/(sizeof(KeyT)+sizeof(ValueT))) : sort_pairs_default_prob_size;

	run_test_over_entropies<KeyT, ValueT>(num_keys, sort_repeat_count, NULL, 0, "sort_pairs_UINT_UINT64");
}

TEST (Sort_Pairs, Entropy_UINT64_UINT) {
	typedef unsigned long long int KeyT;
	typedef unsigned int ValueT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/(sizeof(KeyT)+sizeof(ValueT))) : sort_pairs_default_prob_size;

	run_test_over_entropies<KeyT, ValueT>(num_keys, sort_repeat_count, NULL, 0, "sort_pairs_UINT64_UINT");
}

TEST (Sort_Pairs, Entropy_UINT64_UINT64) {
	typedef unsigned long long int KeyT;
	typedef unsigned long long int ValueT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/(sizeof(KeyT)+sizeof(ValueT))) : sort_pairs_default_prob_size;

	run_test_over_entropies<unsigned long long int, unsigned long long int>(num_keys, sort_repeat_count, NULL, 0, "sort_pairs_UINT64_UINT64");
}


/*****************************
 *** VARIABLE PROBLEM SIZE ***
 *****************************/
TEST (Sort_Pairs, Entropy_UINT_UINT_NUMKEYS) {
	typedef unsigned int KeyT;
	typedef unsigned int ValueT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/(sizeof(KeyT)+sizeof(ValueT))) : sort_pairs_default_prob_size;

	for(long double i=100000;i<num_keys;i*=1.25892541179416721042395410639580060609361740946L){
		run_test_over_entropies<KeyT, ValueT>(i, sort_repeat_count, NULL, 0, "sort_pairs_numpairs_UINT_UINT");
	}
	double i = num_keys;
	run_test_over_entropies<KeyT, ValueT>(i, sort_repeat_count, NULL, 0, "sort_pairs_numpairs_UINT_UINT");
}

TEST (Sort_Pairs, Entropy_UINT64_UINT64_NUMKEYS) {
	typedef unsigned long long int KeyT;
	typedef unsigned long long int ValueT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/(sizeof(KeyT)+sizeof(ValueT))) : sort_pairs_default_prob_size;

	for(long double i=100000;i<num_keys;i*=1.25892541179416721042395410639580060609361740946L){
		run_test_over_entropies<KeyT, ValueT>(i, sort_repeat_count, NULL, 0, "sort_pairs_numpairs_UINT64_UINT64");
	}
	double i = num_keys;
	run_test_over_entropies<KeyT, ValueT>(i, sort_repeat_count, NULL, 0, "sort_pairs_numpairs_UINT64_UINT64");
}

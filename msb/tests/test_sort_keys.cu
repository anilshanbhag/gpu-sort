#include "gtest/gtest.h"
#include "cub/cub.cuh"
#include "benchmark/benchmark.h"
#include "sort/gpu_radix_sort.h"
#include "gpu_helper/gpu_warmup.cuh"
#include "data_gen.h"
#include "cli_args.h"

static CachingDeviceAllocator  g_allocator(true);

template<
	typename KeyT
>
void cub_sort_keys(const KeyT *h_keys, const unsigned int num_keys, KeyT *h_keys_sorted)
{
	// Allocate device memory for input/output
    DoubleBuffer<KeyT> d_keys;
    CubDebugExit( g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(KeyT) * num_keys) );
    CubDebugExit( g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_keys) );

    // Initialize device arrays
	CubDebugExit( cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(KeyT) * num_keys, cudaMemcpyHostToDevice) );

    // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit( DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_keys) );
    CubDebugExit( g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes) );

    // Sort
    BM_START_CUDA_EVENT(cub_sort, cub_sort, NULL);
    CubDebugExit( DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_keys) );
    BM_STOP_CUDA_EVENT(cub_sort, cub_sort, NULL);

    // Copy results for verification. GPU-side part is done.
    CubDebugExit( cudaMemcpy(h_keys_sorted, d_keys.Current(), sizeof(KeyT) * num_keys, cudaMemcpyDeviceToHost) );

    // Cleanup
    if (d_keys.d_buffers[0])
    	CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[0]) );
    if (d_keys.d_buffers[1])
    	CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[1]) );
    if (d_temp_storage)
    	CubDebugExit( g_allocator.DeviceFree(d_temp_storage) );
}

template<
	typename KeyT
>
::testing::AssertionResult verify_sorted_keys(KeyT *h_keys, const KeyT *h_keys_sorted, const unsigned int num_keys)
{
	typedef Traits<KeyT>                        KeyTraits;
	typedef typename KeyTraits::UnsignedBits    UnsignedBits;

	// Sort keys for result verification
	cub_sort_keys<KeyT>(h_keys, num_keys, h_keys);

	// Compare results for verification
	int cmp = memcmp(h_keys, h_keys_sorted, num_keys * sizeof(KeyT));

	// Results are identical
	if(cmp == 0){
		return ::testing::AssertionSuccess();
	}

	// Results mismatch: track down index of mismatch for debugging
	else{
		for(int i = 0; i < num_keys; i++){
			// Do binary comparison to work around NaN != NaN mismatches
			if((reinterpret_cast<UnsignedBits*>(h_keys))[i] != (reinterpret_cast<const UnsignedBits*>(h_keys_sorted))[i]){
				::testing::AssertionResult test_result = ::testing::AssertionFailure();
				test_result << "Mismatch at index " << i << "\n";
				for(int j=(i-100<0?0:i);j<(i+100>num_keys?num_keys:i+100);j++)
					test_result << "i:" << j << " - HEX(CPU vs GPU): " << std::hex << (reinterpret_cast<UnsignedBits*>(h_keys))[j] << std::nouppercase << " - " << std::hex  << (reinterpret_cast<const UnsignedBits*>(h_keys_sorted))[j] << "\n";
				return test_result;
			}
		}
	}
	return ::testing::AssertionFailure();
}

template <
	typename KeyT
>
void test_sort_keys(unsigned int num_keys, int entropy_level, unsigned long long int seed)
{
	typedef Traits<KeyT>                        	KeyTraits;
	typedef typename KeyTraits::UnsignedBits    	UnsignedBits;

	// Allocate memory for data
	KeyT *h_keys_in = (KeyT *)malloc(num_keys*sizeof(*h_keys_in));
	KeyT *h_keys_cpy = (KeyT *)malloc(num_keys*sizeof(*h_keys_cpy));
	KeyT *h_keys_sorted = (KeyT *)malloc(num_keys*sizeof(*h_keys_sorted));

	// Prepare data
	BM_START_CPU_PROFILE(random data, rand_data);
	generate_random_keys(reinterpret_cast<UnsignedBits*>(h_keys_in), num_keys, 0, entropy_level);
	BM_STOP_CPU_PROFILE(random data, rand_data);

	// Copy keys for verification
	memcpy(h_keys_cpy, h_keys_in, num_keys*sizeof(*h_keys_in));

	// Sort
	rdxsrt_warmup_gpu();
	BM_START_CUDA_EVENT(cuda time, sort_total, NULL);
	rdxsrt_unstable_sort_keys<KeyT>(h_keys_in, num_keys, h_keys_sorted);
	BM_STOP_CUDA_EVENT(cuda time, sort_total, NULL);

	// Verification
	BM_START_CPU_PROFILE(cpu verification, cpu_verif);
	ASSERT_TRUE(verify_sorted_keys(h_keys_cpy, h_keys_sorted, num_keys)) << "Test: \n" << " - key count: " << num_keys << "\n - entropy level: " << entropy_level << "\n";
	BM_STOP_CPU_PROFILE(cpu verification, cpu_verif);

	// Free resources
	free(h_keys_in);
	free(h_keys_cpy);
	free(h_keys_sorted);
}


template <
	typename KeyT
>
void run_test_over_entropies(unsigned int num_keys, unsigned int repeat_count = 3, int *entropy_levels = NULL, int num_entropies = 0, const char *profile_name = "sort_keys")
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
		double entropy = 8*sizeof(KeyT) * single_bit_entropy;

		for(int i=0; i<repeat_count; i++){
			printf(" --- SORTKEYS.ENTROPIES (Entropy Lev. (bit entropy): %2d (%6.3f), Iteration: %2d - key_count: %u)---\n", entropy_level, entropy, i, num_keys);

			BM_OPEN_PROFILE(profile_name);
			BM_SET_METRIC_UINT(num_keys, num_keys, num_keys);
			BM_SET_METRIC_INT(entr_and, entr_and, entropy_level);
			BM_SET_METRIC_DOUBLE(bit_entr, bit_entropy, (entropy_level<=0?0.0:entropy));
			test_sort_keys<KeyT>(num_keys, entropy_level, 0);
			BM_CLOSE_PROFILE();
		}
	}
}

/*****************************
 *** CONSTANT PROBLEM SIZE ***
 *****************************/
TEST (Sort_Keys, Entropy_UINT) {
	typedef unsigned int KeyT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/sizeof(KeyT)) : sort_keys_default_prob_size;
	run_test_over_entropies<KeyT>(num_keys, sort_repeat_count, NULL, 0, "sort_keys_UINT");
}

TEST (Sort_Keys, Entropy_UINT64) {
	typedef unsigned long long int KeyT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/sizeof(KeyT)) : sort_keys_default_prob_size;
	run_test_over_entropies<unsigned long long int>(num_keys, sort_repeat_count, NULL, 0, "sort_keys_UINT64");
}

TEST (Sort_Keys, Entropy_DOUBLE) {
	typedef double KeyT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/sizeof(KeyT)) : sort_keys_default_prob_size;
	run_test_over_entropies<KeyT>(num_keys, sort_repeat_count, NULL, 0, "sort_keys_DOUBLE");
}

/*****************************
 *** VARIABLE PROBLEM SIZE ***
 *****************************/
TEST (Sort_Keys, Entropy_UINT_NUMKEYS) {
	typedef unsigned int KeyT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/sizeof(KeyT)) : sort_keys_default_prob_size;

	for(long double i=100000;i<num_keys;i*=1.25892541179416721042395410639580060609361740946L){
		run_test_over_entropies<KeyT>(i, sort_repeat_count, NULL, 0, "sort_keys_numkeys_UINT");
	}
	double i = num_keys;
	run_test_over_entropies<KeyT>(i, sort_repeat_count, NULL, 0, "sort_keys_numkeys_UINT");
}

TEST (Sort_Keys, Entropy_UINT64_NUMKEYS) {
	typedef unsigned long long int KeyT;
	unsigned int num_keys = sort_keys_prob_size_in_mb ? sort_keys_prob_size_in_mb * (1000000/sizeof(KeyT)) : sort_keys_default_prob_size;

	for(long double i=100000;i<num_keys;i*=1.25892541179416721042395410639580060609361740946L){
		run_test_over_entropies<KeyT>((int)i, sort_repeat_count, NULL, 0, "sort_keys_numkeys_UINT64");
	}
	double i = num_keys;
	run_test_over_entropies<KeyT>((int)i, sort_repeat_count, NULL, 0, "sort_keys_numkeys_UINT64");
}


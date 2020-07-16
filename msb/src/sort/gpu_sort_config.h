#ifndef GPU_SORT_CONFIG_H_
#define GPU_SORT_CONFIG_H_

#include <vector>
#include <algorithm>
#include "cub/cub.cuh"
#include "gpu_helper/gpu_helper.cuh"
#include "sort/cuda_radix_sort.h"
/************************************************
 * DEFAULT LOCAL SORT CONFIGS 					*
 ************************************************/

/*** FORWARD DECLARATIONS ***/
template<typename KeyT, typename ValueT>
struct LocalRadixSortConfig;

template<typename KeyT, typename ValueT>
struct LocalSortConfigSet;

#define RDXSRT_LOCAL_SORT_CONFIG(KPT, TPB, ValueT) \
	config = new LocalRadixSortConfig<KeyT, ValueT>(KPT, TPB, do_locrec_radix_sort_keys<KeyT, ValueT, unsigned int, 8, KPT, TPB, false, 4>, do_locrec_radix_sort_keys<KeyT, ValueT, unsigned int, 8, KPT, TPB, true>, do_locrec_radix_sort_keys<KeyT, ValueT, unsigned int, 8, KPT, TPB, false, 5>); \
	local_sort_config_set->AddLocalSortConfig(config);

/*** DEFAULT FALLBACK LOCAL SORT CONFIGURATION SET ***/
template<
	typename KeyT,
	typename ValueT,
	int KEY_SIZE,
	int VALUE_SIZE
>
struct DefaultLocalSortConfigSet {
	static void InitConfigSet(LocalSortConfigSet<KeyT, ValueT> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, ValueT> *config;
		RDXSRT_LOCAL_SORT_CONFIG(5, 64, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, ValueT);
//		RDXSRT_LOCAL_SORT_CONFIG(11, 256, ValueT);
//		RDXSRT_LOCAL_SORT_CONFIG(15, 256, ValueT);
	}
};

/*** [KEYS 4] ***/
template<typename KeyT, int VALUE_SIZE>
struct DefaultLocalSortConfigSet<KeyT, cub::NullType, 4, VALUE_SIZE> {
	static void InitConfigSet(LocalSortConfigSet<KeyT, cub::NullType> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, cub::NullType> *config;
		RDXSRT_LOCAL_SORT_CONFIG(5, 64, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(11, 256, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(15, 256, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(15, 384, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(18, 512, cub::NullType);
	}
};

/*** [KEYS 8] ***/
template<typename KeyT, int VALUE_SIZE>
struct DefaultLocalSortConfigSet<KeyT, cub::NullType, 8, VALUE_SIZE> {
	static void InitConfigSet(LocalSortConfigSet<KeyT, cub::NullType> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, cub::NullType> *config;
		RDXSRT_LOCAL_SORT_CONFIG(5, 64, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(11, 256, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(15, 256, cub::NullType);
		RDXSRT_LOCAL_SORT_CONFIG(11, 384, cub::NullType);
	}
};

/*** [KV-PAIRS 4,4] ***/
template<typename KeyT, typename ValueT>
struct DefaultLocalSortConfigSet<KeyT, ValueT, 4, 4> {
	static void InitConfigSet(LocalSortConfigSet<KeyT, ValueT> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, ValueT> *config;
		RDXSRT_LOCAL_SORT_CONFIG(3, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 64, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 256, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 256, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 384, ValueT);
	}
};

/*** [KV-PAIRS 4,8] ***/
template<typename KeyT, typename ValueT>
struct DefaultLocalSortConfigSet<KeyT, ValueT, 4, 8> {
	static void InitConfigSet(LocalSortConfigSet<KeyT, ValueT> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, ValueT> *config;
		RDXSRT_LOCAL_SORT_CONFIG(3, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 64, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 256, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 256, ValueT);
	}
};

/*** [KV-PAIRS 8,4] ***/
template<typename KeyT, typename ValueT>
struct DefaultLocalSortConfigSet<KeyT, ValueT, 8, 4> {
	static void InitConfigSet(LocalSortConfigSet<KeyT, ValueT> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, ValueT> *config;
		RDXSRT_LOCAL_SORT_CONFIG(3, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 64, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 256, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 256, ValueT);
	}
};

/*** [KV-PAIRS 8,8] ***/
template<typename KeyT, typename ValueT>
struct DefaultLocalSortConfigSet<KeyT, ValueT, 8, 8> {
	static void InitConfigSet(LocalSortConfigSet<KeyT, ValueT> *local_sort_config_set)
	{
		LocalRadixSortConfig<KeyT, ValueT> *config;
		RDXSRT_LOCAL_SORT_CONFIG(3, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 32, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 64, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(7, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 128, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(11, 256, ValueT);
		RDXSRT_LOCAL_SORT_CONFIG(15, 256, ValueT);
	}
};

/***********************************************
 * PARTITIONING CONFIGS 					   *
 ***********************************************/
template <
	int KEY_SIZE, 	// Key size in number of bytes
	int VALUE_SIZE	// Value size in number of bytes (0 for keys only)
>
struct RadixSortConfig
{
    enum { TPB = 256 };
    enum { KPT = 9 };
};

/*** KEY SIZE 4 ***/
template<>
struct RadixSortConfig<4, 0>
{
    enum { TPB = 384 };
    enum { KPT = 18 };
};

template<>
struct RadixSortConfig<4, 4>
{
    enum { TPB = 384 };
    enum { KPT = 18 };
};

template<>
struct RadixSortConfig<4, 8>
{
    enum { TPB = 384 };
    enum { KPT = 13 };
};

/*** KEY SIZE 8 ***/
template<>
struct RadixSortConfig<8, 0>
{
    enum { TPB = 384 };
    enum { KPT = 9 };
};

template<>
struct RadixSortConfig<8, 4>
{
    enum { TPB = 256 };
    enum { KPT = 9 };
};

template<>
struct RadixSortConfig<8, 8>
{
    enum { TPB = 256 };
    enum { KPT = 9 };
};

/**
 * Alias template for mapping the
 */
template<
	typename KeyT,
	typename ValueT
>
using DefaultRadixSortConfig = typename cub::If<cub::Equals<ValueT, cub::NullType>::VALUE, RadixSortConfig<sizeof(KeyT), 0>, RadixSortConfig<sizeof(KeyT), sizeof(ValueT)>>::Type;

/***********************************************
 * LOCAL SORT CONFIGS 						   *
 ***********************************************/
template<typename KeyT, typename ValueT>
using LocalSortKernel = void (*)(KeyT *__restrict__ keys_in, ValueT *__restrict__ values_in, const struct rdxsrt_recsrt_block_info_t<RDXSRT_CFG_MERGE_LOCREC_THRESH> *__restrict__ block_task_infos, const unsigned char byte, KeyT *__restrict__ keys_out, ValueT *__restrict__ values_out);

template<
	typename KeyT,
	typename ValueT
>
struct LocalSortConfig {
	unsigned int tpb;
	unsigned int kpt;
	unsigned int kpb;
	LocalSortConfig():tpb(0),kpt(0),kpb(0){};
	LocalSortConfig(unsigned int kpt, unsigned int tpb)
		:	kpt(kpt),
		 	tpb(tpb),
		 	kpb(kpt*tpb)
	{};
	virtual LocalSortKernel<KeyT, ValueT> GetSortKernel(unsigned char remaining_bits){
		return NULL;
	}
};

template<
	typename KeyT,
	typename ValueT
>
struct LocalRadixSortConfig : LocalSortConfig<KeyT, ValueT> {
	LocalSortKernel<KeyT, ValueT> default_sort_kernel;
	LocalSortKernel<KeyT, ValueT> last_pass_sort_kernel;
	LocalSortKernel<KeyT, ValueT> many_bits_sort_kernel;

	LocalRadixSortConfig(unsigned int kpt, unsigned int tpb, LocalSortKernel<KeyT, ValueT> default_sort_kernel, LocalSortKernel<KeyT, ValueT> last_pass_sort_kernel, LocalSortKernel<KeyT, ValueT> many_bits_sort_kernel)
			: LocalSortConfig<KeyT, ValueT>(kpt, tpb),
			  default_sort_kernel(default_sort_kernel),
			  last_pass_sort_kernel(last_pass_sort_kernel),
			  many_bits_sort_kernel(many_bits_sort_kernel)
		{}

	virtual LocalSortKernel<KeyT, ValueT> GetSortKernel(unsigned char remaining_bits){
		// If only one sorting pass remains for the LSD, use the optimised kernel
		if(remaining_bits == 8 && last_pass_sort_kernel){
			return last_pass_sort_kernel;
		}
		// Using 5 bits per CUB-sorting-pass only pays off if a pass can be saved. Moreover, one pass of 8 bits is done by unstable sorting pass, therefore '-8'
		else if(remaining_bits-8 >= 20 && many_bits_sort_kernel){
			return many_bits_sort_kernel;
		}
		// If there's no optimised kernel available, fall back to the default one
		else{
			return default_sort_kernel;
		}
	}

};

template<
	typename KeyT,
	typename ValueT
>
struct LocalSortConfigSet {
	enum { KEY_SIZE = sizeof(KeyT) };
	enum { VALUE_SIZE = cub::If<
							cub::Equals<ValueT, cub::NullType>::VALUE,
							cub::Int2Type<0>,
							cub::Int2Type<sizeof(ValueT)>
						>::Type::VALUE };

	std::vector< LocalSortConfig<KeyT, ValueT>* > sort_configs;
	int* dev_sort_thresholds;

	static LocalSortConfigSet<KeyT, ValueT> *default_config_set;
	static LocalSortConfigSet<KeyT, ValueT> *GetDefaultConfigSet()
	{
		if(!LocalSortConfigSet<KeyT, ValueT>::default_config_set){
			LocalSortConfigSet<KeyT, ValueT>::default_config_set = new LocalSortConfigSet<KeyT, ValueT>();
			DefaultLocalSortConfigSet<KeyT, ValueT, KEY_SIZE, VALUE_SIZE>::InitConfigSet(LocalSortConfigSet<KeyT, ValueT>::default_config_set);
		}
		return LocalSortConfigSet<KeyT, ValueT>::default_config_set;
	}

	static bool CompareSortConfigs(LocalSortConfig<KeyT, ValueT>* lhs, LocalSortConfig<KeyT, ValueT>* rhs)
	{
	  return lhs->kpb < rhs->kpb;
	}

	LocalSortConfigSet() : dev_sort_thresholds(NULL) {
	}

	void AddLocalSortConfig(LocalSortConfig<KeyT, ValueT> *local_sort_config)
	{
		// TODO throw error when dev_sort_thresholds is not NULL anymore
		sort_configs.push_back(local_sort_config);
	}

	unsigned int num_configs()
	{
		return sort_configs.size();
	}

	unsigned int max_kpb()
	{
		if(sort_configs.size()==0)
			return 0;

		std::sort(sort_configs.begin(), sort_configs.end(), LocalSortConfigSet<KeyT, ValueT>::CompareSortConfigs);
		return sort_configs[sort_configs.size()-1]->kpb;
	}

	int *GetDeviceLocalSortConfigThresholds(cudaStream_t cudaStream = NULL)
	{
		if(!dev_sort_thresholds){
			std::sort(sort_configs.begin(), sort_configs.end(), LocalSortConfigSet<KeyT, ValueT>::CompareSortConfigs);
			for (auto a : sort_configs) {
				std::cout << a->kpb << " \n";
			}
			int *thresholds = new int [sort_configs.size()];
			for(int i=0;i<sort_configs.size();i++)
				thresholds[i] = sort_configs[i]->kpb;
			cudaMalloc(&dev_sort_thresholds, sizeof(*dev_sort_thresholds)*sort_configs.size());
			cudaMemcpyAsync(dev_sort_thresholds, thresholds, sizeof(*dev_sort_thresholds)*sort_configs.size(), cudaMemcpyHostToDevice, cudaStream);
			printf("--- COPYING LOCAL SORT THRESHOLDS FOR KEY SIZE (%d, %d) ---\n", KEY_SIZE, VALUE_SIZE);
		}
		return dev_sort_thresholds;
	}
};


template<typename KeyT,typename ValueT>
LocalSortConfigSet<KeyT, ValueT> *LocalSortConfigSet<KeyT, ValueT>::default_config_set = NULL;

#endif /* GPU_SORT_CONFIG_H_ */

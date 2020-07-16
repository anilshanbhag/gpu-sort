#ifndef DATA_GEN_H_
#define DATA_GEN_H_

#include <curand.h>
#include <stdio.h>
#include "cub/cub.cuh"

template <typename KeyT>
void __global__ bitwise_and_keys(KeyT *keys_in_out, const KeyT *keys_and, const size_t num_keys)
{
	typedef cub::Traits<KeyT>					KeyTraits;
	typedef typename KeyTraits::UnsignedBits	UnsignedBits;

	size_t t_index = blockDim.x * blockIdx.x + threadIdx.x;
	if(t_index < num_keys){
		UnsignedBits tmp_masked = reinterpret_cast<UnsignedBits*>(keys_in_out)[t_index] & reinterpret_cast<const UnsignedBits*>(keys_and)[t_index];
		keys_in_out[t_index] = *reinterpret_cast<KeyT*>(&tmp_masked);
	}
}

template <typename KeyT>
void __global__ mask_keys(KeyT *keys, const unsigned long long int mask, const size_t num_keys)
{
	typedef cub::Traits<KeyT>					KeyTraits;
	typedef typename KeyTraits::UnsignedBits	UnsignedBits;

	size_t t_index = blockDim.x * blockIdx.x + threadIdx.x;
	if(t_index < num_keys){
		UnsignedBits masked_key = reinterpret_cast<UnsignedBits *>(keys)[t_index] & mask;
		keys[t_index] = *reinterpret_cast<KeyT *>(&masked_key);
	}
}

template <typename KeyT>
void dev_generate_uniform_random_keys(KeyT *d_rand_data, const size_t num_keys, const unsigned long long int seed)
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandGenerate(gen, reinterpret_cast<unsigned int*>(d_rand_data), (num_keys*sizeof(KeyT) + sizeof(unsigned int)-1) / sizeof(unsigned int));
	curandDestroyGenerator(gen);
}

template <typename KeyT>
void generate_random_keys(KeyT *keys, const size_t num_keys, const unsigned long long int seed, const int entropy_level = 1, const unsigned long long int mask = 0xFFFFFFFFFFFFFFFFUL)
{
	const unsigned int TPB = 256;
	const unsigned int num_blocks = (num_keys+TPB-1)/TPB;

	KeyT *d_uniform_data;
	KeyT *d_keys_out;
	cudaMalloc((void **)&d_uniform_data, num_keys*sizeof(KeyT));
	cudaMalloc((void **)&d_keys_out, num_keys*sizeof(KeyT));

	// For entropy_level <= 0, we generate cosntant distribution of 0's
	if(entropy_level<1){
		mask_keys<<<num_blocks, TPB>>>(d_keys_out, 0, num_keys);
	}else{
		// Generate uniform key distribution
		dev_generate_uniform_random_keys<KeyT>(d_keys_out, num_keys, seed);

		// Repeatedly apply bitwise-and for reduced entropy
		for(int i=1; i < entropy_level; i++){
			dev_generate_uniform_random_keys<KeyT>(d_uniform_data, num_keys, seed+i*17);
			bitwise_and_keys<KeyT><<<num_blocks, TPB>>>(d_keys_out, d_uniform_data, num_keys);
		}

		// Mask keys
		mask_keys<<<num_blocks, TPB>>>(d_keys_out, mask, num_keys);
	}

	// Copy random data
	cudaMemcpy(keys, d_keys_out, num_keys * sizeof(KeyT), cudaMemcpyDeviceToHost);
	cudaFree(d_uniform_data);
	cudaFree(d_keys_out);
}


template <typename ValueT>
void generate_enumerated_values(ValueT *values, const size_t num_values)
{
	#pragma unroll 8
	for(size_t i=0; i<num_values; i++)
		values[i] = i;
}

#endif /* DATA_GEN_H_ */

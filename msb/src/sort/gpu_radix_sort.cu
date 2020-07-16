#include "benchmark/debug_logger.h"
#include "benchmark/benchmark.h"
#include "gpu_helper/gpu_helper.cuh"
#include "sort/cuda_radix_sort_config.h"
#include "sort/gpu_sort_config.h"
#include "sort/gpu_radix_sort.h"
#include "sort/cuda_radix_sort.h"

void rdxsrt_prepare_histograms(unsigned int *dev_histo, unsigned int num_ext_in_bins, cudaStream_t cstream)
{
	// For every IN-bin that needs to be sorted externally, we need one histogram with each 256 sub-bins
	if(num_ext_in_bins>0)
		cudaMemsetAsync(dev_histo, 0, num_ext_in_bins * 256 * sizeof(*dev_histo), cstream);
}

/** *
 * Takes as input the number of extsrt OUT-bins of the last pass (dev_atomic_ext_srt_counter), along with info on on their unique index, offset and number of elements (dev_extsrt_in_bin_info),
 * copies the information to extsrt_in_bin_info and computes the block assignments accordingly, writing it to extsrt_blck_to_in_bin_assignments.
 *
 * @param dev_atomic_ext_srt_bin_counter		The number of EXTSRT-OUT-bins from the last pass. Equals the number of EXTSRT-IN-bins of this pass.
 * @param dev_extsrt_in_bin_info				Information about the EXTSRT-OUT-bins from the last pass, such as a unique index (bin_idx), the bin's offset within the *keys array (offset) and the number of keys in that bin (block_num_elements).
 * @param extsrt_in_bin_info					Pointer to host side memory to which dev_extsrt_in_bin_info should be copied to.
 * @param extsrt_blck_to_in_bin_assignments		Pointer to memory to which the block assignments that are computed according to dev_extsrt_in_bin_info should be written to.
 * @param max_ext_srt_kpb						The maximum number of keys each extsrt-block will cover
 * @param cstream								The cuda stream to use for copying the device side information of the EXTSRT-bins to the Host
 * @param offst_comp_event						The event to synchronize with, such that it is guaranteed that the information in dev_atomic_ext_srt_bin_counter and dev_extsrt_in_bin_info is final.
 * @return	Returns the number of EXTSRT-bins and the total number of EXTSRT-blocks required to externally radix-sort the EXTSRT-bins.
 */
struct extsrt_assignment_info_t generate_next_pass_block_assignments(const unsigned int *dev_atomic_extsrt_bucket_counter, const SubBucketInfo *dev_extsrt_in_bin_info, SubBucketInfo *extsrt_in_bin_info,  struct rdxsrt_extsrt_block_to_bin_t *extsrt_blck_to_in_bin_assignments, unsigned int max_ext_srt_kpb, cudaStream_t cstream, cudaEvent_t offst_comp_event){
	// Make sure GPU has finished computing the sub-buckets of the last pass
	cudaEventSynchronize(offst_comp_event);

	// Get number of sub-buckets that require another counting sort
	unsigned int num_next_pass_ext_buckets;
	cudaMemcpyAsync(&num_next_pass_ext_buckets, dev_atomic_extsrt_bucket_counter, sizeof(num_next_pass_ext_buckets), cudaMemcpyDeviceToHost, cstream);
//	printf("Num next pass buckets: %u\n", num_next_pass_ext_buckets);
	cudaStreamSynchronize(cstream); // TODO think of giving it it's own stream, if that makes sense: look at the dependency - we want to interleave CPU preparation and copy, in best case with local sort

	// Get info (.offset: sub-bucket offset, .block_num_elements: total number of sub-bucket's keys) on the sub-buckets that require another counting sort
	if(num_next_pass_ext_buckets==0){
		struct extsrt_assignment_info_t tmp;
		tmp.num_ext_bins = 0;
		tmp.num_extsrt_block_count = 0;
		tmp.num_extsrt_remainder_block_count = 0;
		tmp.extsrt_full_block_offset = 0;
		return tmp;
	}
	cudaMemcpyAsync(extsrt_in_bin_info, dev_extsrt_in_bin_info, num_next_pass_ext_buckets * sizeof(*extsrt_in_bin_info), cudaMemcpyDeviceToHost, cstream);
	cudaStreamSynchronize(cstream);

	// Counting the required number of full blocks for the counting sorts of the coming pass
	unsigned int num_next_pass_ext_block_count = 0;
	// Counting the required number of remainder blocks (not-full) for the counting sorts of the coming pass
	unsigned int num_extsrt_remainder_block_count = 0;

	// The first num_next_pass_ext_buckets assignments are reserved for the up to num_next_pass_ext_buckets remainder blocks, such that in memory we have:
	// <---REMAINDER BLOCK ASSIGNMENTS---> <------------ num_next_pass_ext_block_count FULL BLOCK ASSIGNMENTS ------------------->
	// [0,1,..,num_next_pass_ext_buckets-1,num_next_pass_ext_buckets,...,num_next_pass_ext_block_count+num_next_pass_ext_buckets-1]
	unsigned int full_block_offset = num_next_pass_ext_buckets;

	// Prepare one assignment for each single block involved in processing the counting sort for one of the buckets
//	START_CPU_TIMER(1, bm_cpu_genass);
//	printf("START\n");
	for(unsigned int i=0; i < num_next_pass_ext_buckets; i++){
		SubBucketInfo tmp;
		tmp = extsrt_in_bin_info[i];
		unsigned int bin_offset = tmp.offset;
//		printf("next pass bucket %5d (id, offset, size): %u, %u, %u\n", i, tmp.bin_idx, tmp.offset, tmp.block_num_elements);
//		printf("%u, %u, %u\n", tmp.bin_idx, tmp.offset, tmp.block_num_elements);
		for(unsigned int j=0; tmp.block_num_elements > 0; j++){
			// Another full block
			if(tmp.block_num_elements >= max_ext_srt_kpb){
				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].bin_idx = tmp.bin_idx;
				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].offset = tmp.offset;
				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].bin_start_offset = bin_offset;
				extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].block_num_elements = max_ext_srt_kpb;

				tmp.offset += extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].block_num_elements;
				tmp.block_num_elements -= extsrt_blck_to_in_bin_assignments[full_block_offset+num_next_pass_ext_block_count].block_num_elements;
				num_next_pass_ext_block_count++;
			}
			// The remainder block of that bucket
			else{
				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].bin_idx = tmp.bin_idx;
				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].offset = tmp.offset;
				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].bin_start_offset = bin_offset;
				extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].block_num_elements = tmp.block_num_elements > max_ext_srt_kpb ? max_ext_srt_kpb : tmp.block_num_elements;

				tmp.offset += extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].block_num_elements;
				tmp.block_num_elements -= extsrt_blck_to_in_bin_assignments[num_extsrt_remainder_block_count].block_num_elements;
				num_extsrt_remainder_block_count++;
			}
		}
	}
//	printf("END\n");
//	STOP_CPU_TIMER(bm_cpu_genass, "CPU prepare block assignments: ");

	struct extsrt_assignment_info_t tmp;
	tmp.num_ext_bins = num_next_pass_ext_buckets;
	tmp.num_extsrt_block_count = num_next_pass_ext_block_count;
	tmp.num_extsrt_remainder_block_count = num_extsrt_remainder_block_count;
	tmp.extsrt_full_block_offset = full_block_offset;
	return tmp;
}




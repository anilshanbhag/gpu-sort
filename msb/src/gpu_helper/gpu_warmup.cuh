#ifndef GPU_WARMUP_CUH_
#define GPU_WARMUP_CUH_

template <typename T>
__global__ void gpu_warm_up_code_kernel(unsigned int *fake_out_ptr)
{
	__shared__ T common;
	T tmp = blockDim.x * blockIdx.x * ((13*threadIdx.x)%17)+15;
	tmp = atomicAdd(&common, tmp);
	if(tmp == 4000000001)
		*fake_out_ptr = 17;
}

__inline void rdxsrt_warmup_gpu()
{
	unsigned int *dev_warmup_ctr;
	cudaMalloc(&dev_warmup_ctr, sizeof(*dev_warmup_ctr));
	gpu_warm_up_code_kernel<unsigned int><<<16*1024*1024,1024>>>(dev_warmup_ctr);
	cudaFree(dev_warmup_ctr);
}

#endif /* GPU_WARMUP_CUH_ */

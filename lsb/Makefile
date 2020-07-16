NVCC = nvcc

SM_TARGETS 	= -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
SM_DEF 		= -DSM520

#SM_TARGETS 	= -gencode=arch=compute_70,code=\"sm_70,compute_70\" 
#SM_DEF 		= -DSM700

GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
#GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS   := $(GENCODE_SM50)

NVCCFLAGS = --std=c++11 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 

CUB_DIR = ./cub/
INCLUDES = -I$(CUB_DIR) -I$(CUB_DIR)test

sort: sort.cu
	$(NVCC) $(NVCCFLAGS) $(SM_TARGETS) $(INCLUDES) -lcurand $^ -o $@	

setup:
	if [ ! -d "cub"  ]; then \
		wget https://github.com/NVlabs/cub/archive/1.6.4.zip; \
		unzip 1.6.4.zip; \
		mv cub-1.6.4 cub; \
		rm 1.6.4.zip; \
	fi



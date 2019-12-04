all: nms_kernel.cu nms_kernel.h
	/usr/local/cuda/bin/nvcc -c -o nms.cu.o nms_kernel.cu --compiler-options -fPIC -gencode arch=compute_52,code=sm_52
clean:
	rm nms.cu.o

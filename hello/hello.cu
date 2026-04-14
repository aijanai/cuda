#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void test01(){
    printf("Block ID %d, Thread ID %d, Warp ID %d\n", blockIdx.x, threadIdx.x, threadIdx.x/32);
}

int main(){
    test01 <<<2, 64>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceReset();
    return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>


int main(){
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    printf("CUDA devices: %d\n", num_devices);
    int max_threads_per_sm;

    for (int i=0; i<num_devices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d\n", i);
        printf("Name: %s\n", prop.name);
        printf("SM: %d\n", prop.multiProcessorCount);
        printf("Block/SM: %d\n", prop.maxBlocksPerMultiProcessor);

        cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, i);
        printf("Threads/SM: %d\n",max_threads_per_sm);
        assert(prop.maxThreadsPerMultiProcessor==max_threads_per_sm);
 
        printf("Warps/SM: %d\n", prop.maxThreadsPerMultiProcessor/32);
        printf("Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("Warp/Block: %d\n", prop.maxThreadsPerBlock/32);
        printf("Shared mem/Block: %luKB\n", prop.sharedMemPerBlock/1024);
        printf("Total Memory: %luGB\n", prop.totalGlobalMem/(1024*1024*1024));
        printf("Capability: %d.%d\n", prop.major, prop.minor);
    }
    
    cudaDeviceReset();
    return 0;
}
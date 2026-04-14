#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


int main(){
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    printf("CUDA devices: %d\n", num_devices);
    for (int i=0; i<num_devices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d\n", i);
        printf("Name: %s\n", prop.name);
        printf("SM: %d\n", prop.multiProcessorCount);
        printf("Block/SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("Shared mem/Block: %luKB\n", prop.sharedMemPerBlock/1024);
        printf("Total Memory: %luGB\n", prop.totalGlobalMem/(1024*1024*1024));
        printf("Capability: %d.%d\n", prop.major, prop.minor);
    }
    
    cudaDeviceReset();
    return 0;
}
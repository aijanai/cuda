#include <cuda.h>
#include <cassert>
#include "funcs.h"
#include "kernel.cuh"

unsigned long computeSize(unsigned long n, unsigned int threads_per_block, int ratio=1){
    return (n+ratio*threads_per_block-1)/(ratio*threads_per_block);
}
unsigned long computeNumBlocks(unsigned long n, unsigned int threads_per_block, int ratio=1){
    return computeSize(n, threads_per_block,ratio);
}
int computeBlocksize(unsigned long n, unsigned int blocks){
    return computeSize(n,blocks,1);
}

int main(int argc, char** argv){

    if(argc<3){
        printf("Usage: %s <len> <blocksize>\n", argv[0]);
        return -2;
    }
    unsigned long n=atol(argv[1]); // n
    int threads_per_block=atoi(argv[2]);

    unsigned long blocks=computeNumBlocks(n,threads_per_block,2); //grisize, 2 elements/thread
    int blocksize=computeBlocksize(n,blocks); //blocksize, how many threads per block

    printf("array len is %lu, block size (threads/block) is %d, %d elements/block -> grid size is %lu\n", n, threads_per_block,blocksize, blocks);

    size_t SIZE=n*sizeof(unsigned long);

    unsigned long *a, *b;
    unsigned long *ga;

    // alloc on CPU
    a = new unsigned long[n];
    b = new unsigned long[n];
    #ifdef DEBUG
    unsigned long *c;
    if(DEBUG>0){
       c = new unsigned long[n];
    }
    #endif

    // fill in numbers
    for(unsigned long i=0; i<n; i++){
        a[i]=1;
        b[i]=a[i];
    }

    #ifdef DEBUG
    if(DEBUG>1){
    printf("a: \n");
    printArray<unsigned long>(a,n);
    printf("\n");
    }
    #endif

    cudaError_t err;
    cudaEvent_t start,stop, func_start, func_stop, mem_copied, check;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&func_start);
    cudaEventCreate(&func_stop);
    cudaEventCreate(&mem_copied);
    cudaEventCreate(&check);

    cudaEventRecord(start);
    // alloc on GPU
    err=cudaMalloc((void**) &ga, SIZE);
    if (err != cudaSuccess){
        printf("CUDA error: ga malloc %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // copy from CPU to GPU
    cudaEventRecord(mem_copied);
    err=cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("CUDA error: ga memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaEventRecord(func_start);
    // exec kernel

    int reduced_n=n;
    int reduced_blocks=blocks;
    
    #ifdef DEBUG
    bool first_run=true;
    #endif

    while(true){
        // kernel N
        printf("\n");
        #ifdef DEBUG
        if(DEBUG>0){
            // copy partials
            err=cudaMemcpy(c, ga, reduced_n*sizeof(unsigned long), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess){
                printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
                return -1;
            }

            if(DEBUG>1 && !first_run){
                printf("partial sums:\n");
                printArray<unsigned long>(c,n);
            }
        }
        #endif
        
        #ifdef DEBUG
        if(DEBUG>0){
            printf("sum of first %d nums: %lu\n",reduced_n,sumArrayLinear<unsigned long>(c,reduced_n));
        }
        #endif

        printf("running over length %d with %d blocks\n", reduced_n, reduced_blocks);

        // exec kernel N
        reduce<unsigned long, 256><<<reduced_blocks, threads_per_block>>>(ga, reduced_n);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess){
            printf("CUDA error: dev sync %s\n", cudaGetErrorString(err));
            return -1;
        }
        if(reduced_blocks==1){
            break;
        }
        reduced_n=reduced_blocks;
        reduced_blocks=computeNumBlocks(reduced_n,threads_per_block);

        #ifdef DEBUG
        first_run=false;
        #endif
    }

    // wait for finish
    cudaEventRecord(func_stop);
    // copy from GPU to CPU
    err=cudaMemcpy(a, ga, 1*sizeof(unsigned long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    #ifdef DEBUG
    if(DEBUG>1){
        printf("dump first %d records: \n",reduced_n);
        for(int i=0; i<reduced_n; i++){
            printf(fmt<unsigned long>(),a[i]);
        }
        printf("\n");
    }
    #endif

    printf("running check...");
    auto sum=sumArrayLinear<unsigned long>(b,n);
    if(sum!=a[0]){
        printf("Expected sum: %lu, got %lu\n", sum, a[0]);
        assert(sum==a[0]);
    }

    printf("\nRESULT: %lu\n\n", a[0]);

    float overall_exec, func_exec, cuda_malloc, mem_copy, mem_copy_back;
    cudaEventElapsedTime(&overall_exec, start, stop);
    cudaEventElapsedTime(&cuda_malloc, start, mem_copied);
    cudaEventElapsedTime(&mem_copy, mem_copied, func_start);
    cudaEventElapsedTime(&func_exec, func_start, func_stop);
    cudaEventElapsedTime(&mem_copy_back, func_stop, stop);
    printf("Took %f ms (cuda malloc %f ms -> mem copy %f ms -> func exec %f ms -> mem copy back %f ms)\n", overall_exec, cuda_malloc, mem_copy, func_exec, mem_copy_back);

    cudaFree(ga);
    delete[] a;
    delete[] b;
    #ifdef DEBUG
    if(DEBUG>0){
        delete[] c;
    }
    #endif

    cudaDeviceReset();
    return 0;
}
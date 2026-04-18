#include <cuda.h>
#include <cassert>
#include "funcs.h"
#include "kernel.cuh"


int main(int argc, char** argv){

    if(argc<3){
        printf("Usage: %s <len> <blocksize>\n", argv[0]);
        return -2;
    }
    unsigned long n=atol(argv[1]); // n
    unsigned int threads_per_block=atoi(argv[2]); // blocksize
    unsigned long blocks=(n+threads_per_block-1)/threads_per_block; //grisize

    printf("array len is %lu, block size is %d, grid size is %lu\n", n, threads_per_block, blocks);

    size_t SIZE=n*sizeof(unsigned long);

    unsigned long *a, *b;
    unsigned long *ga;

    // alloc on CPU
    a = new unsigned long[n];
    b = new unsigned long[n];
    #ifdef DEBUG
    unsigned long *c;
    if(DEBUG>1){
       c = new unsigned long[n];
    }
    #endif

    // fill in numbers
    for(unsigned long i=0; i<n; i++){
        a[i]=i;
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
    reduce<unsigned long><<<blocks, threads_per_block>>>(ga, n);

    // wait for finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: dev sync %s\n", cudaGetErrorString(err));
        return -1;
    }
   
    #ifdef DEBUG
    if(DEBUG>1){
        // copy partials
        err=cudaMemcpy(c, ga, n, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
            return -1;
        }

        printf("partial sums:\n");
        printArray<unsigned long>(c,n);
    }
    #endif

    int reduced_n=n;
    int reduced_blocks=blocks;
    if(blocks>1){
        do{
            // kernel N
            reduced_n=reduced_blocks;
            reduced_blocks=(reduced_blocks-1+threads_per_block)/threads_per_block;

            printf("\n");
            #ifdef DEBUG
            if(DEBUG>1){
                printf("sum of first %d nums: %lu\n",reduced_n,sumArrayLinear<unsigned long>(c,reduced_n));
            }
            #endif
            
            printf("re-running over reduced length %d with reduced blocks: %d\n", reduced_n, reduced_blocks);
 
            // exec kernel N
            reduce<unsigned long><<<reduced_blocks, threads_per_block>>>(ga, reduced_n);

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess){
                printf("CUDA error: dev sync %s\n", cudaGetErrorString(err));
                return -1;
            }
        }while(reduced_blocks!=1);
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
    free(a);
    free(b);
    #ifdef DEBUG
    if(DEBUG>1){
        free(c);
    }
    #endif

    cudaDeviceReset();
    return 0;
}
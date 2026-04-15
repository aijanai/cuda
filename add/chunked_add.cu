#include <cuda.h>
#include "funcs.h"
#include <cassert>


void printArray(int* a, int n){
    for(int i=0; i<n; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

int main(){
    unsigned long long n=5000000000;
    unsigned long chunk_size=1024*1024*256;


    int threads_per_block=1024;
    int blocks=(chunk_size+threads_per_block)/threads_per_block;

    size_t SIZE=chunk_size*sizeof(int);

    int *a, *b, *c;
    int *ga, *gb, *gc;

    // alloc on CPU
    a = (int*) malloc(SIZE);
    b = (int*) malloc(SIZE);
    c = (int*) malloc(SIZE);

    // alloc on GPU
    cudaMalloc((void**) &ga, SIZE);
    cudaMalloc((void**) &gb, SIZE);
    cudaMalloc((void**) &gc, SIZE);
    
    cudaEvent_t start,stop, func_start, func_stop, mem_copied;

    float overall_exec, func_exec, cuda_malloc, mem_copy, mem_copy_back;
    cudaError_t err;
    
    unsigned long processed=0;
    for(unsigned long offset=0; offset<n; offset+=chunk_size){

        if (offset+chunk_size> n){
            chunk_size=n-offset;
        }
        printf("Offset: %lu, chunk size: %lu, processed: %lu\n", offset, chunk_size, processed);
    
        // fill in numbers
        for(int i=0; i<chunk_size; i++){
            a[i]=i;
            b[i]=chunk_size-i;
        }

        #ifdef DEBUG
        printf("a: \n");
        printArray(a,n);
        printf("b: \n");
        printArray(b,n);
        #endif
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&func_start);
        cudaEventCreate(&func_stop);
        cudaEventCreate(&mem_copied);

        cudaEventRecord(start);
        // copy from CPU to GPU
        cudaEventRecord(mem_copied);
        cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(gb, b, SIZE, cudaMemcpyHostToDevice);

        cudaEventRecord(func_start);
        // exec kernel
        add<<<blocks, threads_per_block>>>(ga,gb,gc, chunk_size);
        cudaEventRecord(func_stop);

        // wait for finish
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess){
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        // copy from GPU to CPU
        cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&overall_exec, start, stop);
        cudaEventElapsedTime(&cuda_malloc, start, mem_copied);
        cudaEventElapsedTime(&mem_copy, mem_copied, func_start);
        cudaEventElapsedTime(&func_exec, func_start, func_stop);
        cudaEventElapsedTime(&mem_copy_back, func_stop, stop);
        printf("Took %f ms (cuda malloc %f ms -> mem copy %f ms -> func exec %f ms -> mem copy back %f ms)\n", overall_exec, cuda_malloc, mem_copy, func_exec, mem_copy_back);

        #ifdef DEBUG
        printf("c: \n");
        printArray(c,n);
        #endif
    
        for(int i=0; i<chunk_size; i++){
            assert(c[i]==a[i]+b[i]);
        }
        processed+=chunk_size;
    }

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);
    free(a);
    free(b);
    free(c);

    cudaDeviceReset();
    return 0;
}
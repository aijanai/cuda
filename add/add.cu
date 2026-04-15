#include <cuda.h>
#include "funcs.h"


void printArray(int* a, int n){
    for(int i=0; i<n; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

int main(){
    int n=1024*400*1024;
    int blocks=1024;
    int threads_per_block=n/blocks;

    int SIZE=n*sizeof(int);

    int *a, *b, *c;
    int *ga, *gb, *gc;

    // alloc on CPU
    a = (int*) malloc(SIZE);
    b = (int*) malloc(SIZE);
    c = (int*) malloc(SIZE);

    // fill in numbers
    for(int i=0; i<n; i++){
        a[i]=i;
        b[i]=n-i;
    }

    #ifdef DEBUG
    printf("a: \n");
    printArray(a,n);
    printf("b: \n");
    printArray(b,n);
    #endif

    cudaEvent_t start,stop, func_start, func_stop, mem_copied;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&func_start);
    cudaEventCreate(&func_stop);
    cudaEventCreate(&mem_copied);

    cudaEventRecord(start);
    // alloc on GPU
    cudaMalloc((void**) &ga, SIZE);
    cudaMalloc((void**) &gb, SIZE);
    cudaMalloc((void**) &gc, SIZE);

    // copy from CPU to GPU
    cudaEventRecord(mem_copied);
    cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gb, b, SIZE, cudaMemcpyHostToDevice);
    
    cudaEventRecord(func_start);
    // exec kernel
    add<<<blocks, threads_per_block>>>(ga,gb,gc);
    cudaEventRecord(func_stop);

    // wait for finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
   
    // copy from GPU to CPU
    cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float overall_exec, func_exec, cuda_malloc, mem_copy, mem_copy_back;
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

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);
    free(a);
    free(b);
    free(c);

    cudaDeviceReset();
    return 0;
}
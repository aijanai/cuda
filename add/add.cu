#include <cuda.h>
#include <cassert>
#include "funcs.h"


void printArray(int* a, int n){
    for(int i=0; i<n; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

int main(int argc, char** argv){

    if(argc<2){
        printf("Usage: %s <blocks>\n", argv[0]);
        return -2;
    }
    unsigned long n=1024LL*1024*400;
    int threads_per_block=atoi(argv[1]);
    unsigned long blocks=(n+threads_per_block-1)/threads_per_block;

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
    err=cudaMalloc((void**) &gb, SIZE);
    if (err != cudaSuccess){
        printf("CUDA error: gb malloc %s\n", cudaGetErrorString(err));
        return -1;
    }
    err=cudaMalloc((void**) &gc, SIZE);
    if (err != cudaSuccess){
        printf("CUDA error: gc malloc %s\n", cudaGetErrorString(err));
        return -1;
    }

    // copy from CPU to GPU
    cudaEventRecord(mem_copied);
    err=cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("CUDA error: ga memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }
    err=cudaMemcpy(gb, b, SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("CUDA error: gb memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaEventRecord(func_start);
    // exec kernel
    add<<<blocks, threads_per_block>>>(ga,gb,gc, n);
    cudaEventRecord(func_stop);

    // wait for finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: dev sync %s\n", cudaGetErrorString(err));
        return -1;
    }
   
    // copy from GPU to CPU
    err=cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    for(int i=0; i<n; i++){
        if(c[i]!=a[i]+b[i]){
            printf("%d != %d + %d\n", c[i], a[i], b[i]);
            assert(c[i]==a[i]+b[i]);
        }
    }

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
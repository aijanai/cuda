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
    //int blocks=1024;
    //int threads_per_block=n/blocks;

    int SIZE=n*sizeof(int);

    int *a, *b, *c;
    //int *ga, *gb, *gc;

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

    // alloc on GPU
    //cudaMalloc((void**) &ga, SIZE);
    //cudaMalloc((void**) &gb, SIZE);
    //cudaMalloc((void**) &gc, SIZE);

    // copy from CPU to GPU
    //cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
    //cudaMemcpy(gb, b, SIZE, cudaMemcpyHostToDevice);
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // exec kernel
    cudaEventRecord(start);
    for(int i=0; i<n; i++){
        add_impl(a,b,c,i);
    }
    cudaEventRecord(stop);

    // wait for finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
   
    // copy from GPU to CPU
    //cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float millisec=0;
    cudaEventElapsedTime(&millisec, start, stop);
    printf("Took %f ms\n", millisec);
    
    #ifdef DEBUG
    printf("c: \n");
    printArray(c,n);
    #endif

    //cudaFree(ga);
    //cudaFree(gb);
    //cudaFree(gc);
    free(a);
    free(b);
    free(c);

    cudaDeviceReset();
    return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void add(int* a, int* b, int* c, int n){
    int i=threadIdx.x;
    c[i]=a[i]+b[i];
}

void printArray(int* a, int n){
    for(int i=0; i<n; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

int main(){
    int n=1024;

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

    printf("a: \n");
    printArray(a,n);
    printf("b: \n");
    printArray(b,n);

    // alloc on GPU
    cudaMalloc((void**) &ga, SIZE);
    cudaMalloc((void**) &gb, SIZE);
    cudaMalloc((void**) &gc, SIZE);

    // copy from CPU to GPU
    cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gb, b, SIZE, cudaMemcpyHostToDevice);
    
    // exec kernel
    add<<<1, n>>>(ga,gb,gc,n);

    // wait for finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
   
    // copy from GPU to CPU
    cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);

    printf("c: \n");
    printArray(c,n);

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);
    free(a);
    free(b);
    free(c);

    cudaDeviceReset();
    return 0;
}
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cuda.h>
#include <cassert>


__global__ void add(int* a, int* b, int* c, int n){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n){
        c[i]=a[i]+b[i];
    }
}

int main(){
    unsigned long n=1024*1024*400;
    int threads_per_block=1024;
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
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // exec kernel
    cudaEventRecord(start);
    add<<<blocks, threads_per_block>>>(ga,gb,gc,n);
    cudaEventRecord(stop);

    // wait for finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
   
    // copy from GPU to CPU
    err=cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaEventSynchronize(stop);
    for(int i=0; i<n; i++){
        assert(c[i]==a[i]+b[i]);
        if (i>0){
            assert(c[i]==c[i-1]);
        }
    }
    float millisec=0;
    cudaEventElapsedTime(&millisec, start, stop);
    printf("Took %f ms\n", millisec);

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

void printArray(int* a, int n){
    for(int i=0; i<n; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

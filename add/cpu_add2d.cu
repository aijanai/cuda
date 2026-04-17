#include <cuda.h>
#include <cassert>
#include "funcs.h"


int main(int argc, char** argv){

    if(argc<2){
        printf("Usage: %s <blocks>\n", argv[0]);
        return -2;
    }
    unsigned int n=atoi(argv[1]);
    unsigned int m=atoi(argv[1]);
    dim3 block_dim(32,32);
    dim3 grid_dim((n+block_dim.x-1)/block_dim.x,(m+block_dim.y-1)/block_dim.y);

    size_t SIZE=n*m*sizeof(int);

    int *a, *b, *c;
    //int *ga, *gb, *gc;

    // alloc on CPU
    a = (int*) malloc(SIZE);
    b = (int*) malloc(SIZE);
    c = (int*) malloc(SIZE);

    // fill in numbers
    int k;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            k=j+n*i;
            a[k]=k;
            b[k]=n*m-k;
        }
    }

    #ifdef DEBUG
    printf("a: \n");
    printMatrix(a,n,m);
    printf("b: \n");
    printMatrix(b,n,m);
    #endif

    //cudaError_t err;
    cudaEvent_t start,stop, func_start, func_stop, mem_copied, check;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&func_start);
    cudaEventCreate(&func_stop);
    cudaEventCreate(&mem_copied);
    cudaEventCreate(&check);

    cudaEventRecord(start);
    // alloc on GPU
    /*err=cudaMalloc((void**) &ga, SIZE);
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
*/    
    cudaEventRecord(func_start);
    // exec kernel
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            k=j+n*i;
            add_impl(a,b,c,k);
        }
    }
    cudaEventRecord(func_stop);

    /*
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
*/
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            k=j+n*i;
            if(c[k]!=a[k]+b[k]){
                printf("%d != %d + %d\n", c[k], a[k], b[k]);
                assert(c[k]==a[k]+b[k]);
            }
        }
    }

    float overall_exec;
    cudaEventElapsedTime(&overall_exec, start, stop);
    printf("Took %f ms (func exec only) \n", overall_exec);

    #ifdef DEBUG
    printf("c: \n");
    printMatrix(c,n,m);
    #endif

 //   cudaFree(ga);
   // cudaFree(gb);
    //cudaFree(gc);
    free(a);
    free(b);
    free(c);

    cudaDeviceReset();
    return 0;
}
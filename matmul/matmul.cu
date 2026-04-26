#include <cuda.h>
#include <cassert>
#include "funcs.h"
#include "kernel.cuh"


int main(int argc, char** argv){

    if(argc<1){
        printf("Usage: %s <len>\n", argv[0]);
        return -2;
    }

    int n=atoi(argv[1]);
    
    cudaEvent_t start, stop, kernel_start, kernel_stop, cpu_start, cpu_stop, memcpy_start, memcpy_stop, memcpyback_start, memcpyback_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);
    cudaEventCreate(&memcpy_start);
    cudaEventCreate(&memcpy_stop);
    cudaEventCreate(&memcpyback_start);
    cudaEventCreate(&memcpyback_stop);

    cudaEventRecord(start);

    auto* a = initializeMatrix<int>(n,n);
    auto* b = initializeMatrix<int>(n,n);
    auto* c = initializeMatrix<int>(n,n);

    int *ga, *gb, *gc;

    size_t SIZE=getByteSize<int>(n,n);

    cudaMalloc((void**) &ga, SIZE);
    cudaMalloc((void**) &gb, SIZE);
    cudaMalloc((void**) &gc, SIZE);

    for(int i=0; i<n*n; i++){
        a[i]=1;
        b[i]=a[i];
        c[i]=0;
    }

    //b=initializeIdentity<int>(n);

    #ifdef DEBUG
    if(DEBUG>2){
        printf("a:\n");
        printMatrix(a,n,n);
        printf("b:\n");
        printMatrix(b,n,n);
    }
    #endif

    cudaEventRecord(memcpy_start);
    cudaMemcpy(ga,a,SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(gb,b,SIZE,cudaMemcpyHostToDevice);
    cudaEventRecord(memcpy_stop);

    int tile_size=16;
    dim3 blocksize(tile_size,tile_size);
    dim3 grid_size((n+blocksize.x-1)/tile_size, (n+blocksize.y-1)/tile_size);

    printf("Blocksize (%d,%d), gridsize (%d,%d)\n", blocksize.x, blocksize.y, grid_size.x, grid_size.y);
    printf("Running kernel\n");
    cudaEventRecord(kernel_start);
    matmulnaive<<<grid_size,blocksize>>>(ga,gb,gc,n);
    cudaEventRecord(kernel_stop);

    cudaError_t err;
    err=cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA error: dev sync %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaEventRecord(memcpyback_start);
    cudaMemcpy(c,gc,SIZE,cudaMemcpyDeviceToHost);
    cudaEventRecord(memcpyback_stop);

    printf("sum of a input elements: %lld\n",sumMatrix<int, long long>(a,n,n));
    printf("sum of b input elements: %lld\n",sumMatrix<int, long long>(b,n,n));
    printf("sum of c (GPU run) elements: %lld\n",sumMatrix<int, long long>(c,n,n));
    
    cudaEventRecord(cpu_start);
    #ifdef DEBUG
    if(DEBUG>2){
        auto* d = initializeMatrix<int>(n,n);
        printf("Checking for CPU matmul\n");
        cpu_matmul(a,b,d,n);
        printf("sum of d (CPU check) elements: %lld\n",sumMatrix<int, long long>(d,n,n));
        delete[] d;
    }
    #endif
    cudaEventRecord(cpu_stop);
    
    #ifdef DEBUG
    if(DEBUG>2){
        printMatrix(c,n,n);
    }
    #endif

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float overall_exec, kernel_exec, cpu_exec, mem_copy, mem_copy_back;
    err=cudaEventElapsedTime(&overall_exec, start, stop);
    if(err!=cudaSuccess){
        printf("CUDA error: elapsed time %s\n", cudaGetErrorString(err));
        return 3;
    }
    cudaEventElapsedTime(&kernel_exec, kernel_start, kernel_stop);
    cudaEventElapsedTime(&mem_copy, memcpy_start, memcpy_stop);
    cudaEventElapsedTime(&mem_copy_back, memcpyback_start, memcpyback_stop);
    cudaEventElapsedTime(&cpu_exec, cpu_start, cpu_stop);
    printf("Took %f ms (mem copy %f ms -> kern exec %f ms -> mem copy back %f ms + cpu exec %f)\n", overall_exec, mem_copy, kernel_exec, mem_copy_back, cpu_exec);

    //assert(compareMatricesLinear<int>(c,d,n*n));

    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);

    cudaDeviceReset();
    return 0;
}
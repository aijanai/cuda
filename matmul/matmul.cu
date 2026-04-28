#include <cuda.h>
#include <cassert>
#include "funcs.h"
#include "kernel.cuh"
#include <random>

using namespace nvcuda::wmma;

int main(int argc, char** argv){

    if(argc<1){
        printf("Usage: %s <len>\n", argv[0]);
        return -2;
    }

    int n=atoi(argv[1]);
    int tile_size=16;
    
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

    auto* a = initializeMatrix<half>(n,n);
    auto* b = initializeMatrix<half>(n,n);
    auto* c = initializeMatrix<float>(n,n);

    half *ga, *gb;
    float *gc;

    size_t SIZE_half=getByteSize<half>(n,n);
    size_t SIZE=getByteSize<float>(n,n);

    cudaMalloc((void**) &ga, SIZE_half);
    cudaMalloc((void**) &gb, SIZE_half);
    cudaMalloc((void**) &gc, SIZE);

    for(int i=0; i<n*n; i++){
        a[i]=__float2half((float)rand()/RAND_MAX);
        b[i]=a[i];
        c[i]=0;
    }

    //b=initializeIdentity<int>(n);
    assert(compareMatrices<half>(a,b,n,n));

    #ifdef DEBUG
    if(DEBUG>2){
        printf("a:\n");
        printMatrix<half,float>(a,n,n,__half2float);
        printf("b:\n");
        printMatrix<half,float>(b,n,n,__half2float);
    }
    #endif

    cudaEventRecord(memcpy_start);
    cudaMemcpy(ga,a,SIZE_half,cudaMemcpyHostToDevice);
    cudaMemcpy(gb,b,SIZE_half,cudaMemcpyHostToDevice);
    cudaEventRecord(memcpy_stop);

    int tiles_per_dim=n/tile_size;
    int tiles=tiles_per_dim*tiles_per_dim;
    printf("tiles per dim: %d\n", tiles_per_dim);
    int warps_per_block=128/32;
    int blocks=(tiles * warps_per_block -1)/warps_per_block;

    dim3 blocksize(128);
    dim3 grid_size(blocks);

    printf("Blocksize (%d,%d), gridsize (%d,%d)\n", blocksize.x, blocksize.y, grid_size.x, grid_size.y);
    printf("Running kernel\n");
    cudaEventRecord(kernel_start);
    matmultensor<half,float,16><<<grid_size,blocksize>>>(ga,gb,gc,n);
    //matmulvector<float,float4,16><<<grid_size,blocksize>>>(ga,gb,gc,n);
    //matmulnaive<float><<<grid_size,blocksize>>>(ga,gb,gc,n);
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

    printf("sum of a input elements: %lf\n",sumMatrix<half, float>(a,n,n,__half2float));
    printf("sum of b input elements: %lf\n",sumMatrix<half, float>(b,n,n,__half2float));
    printf("sum of c (GPU run) elements: %lf\n",sumMatrix<float, double>(c,n,n,convert<float,double>));
    
    cudaEventRecord(cpu_start);
    #ifdef DEBUG
    if(DEBUG>1){
        auto* d = initializeMatrix<float>(n,n);
        printf("Checking for CPU matmul\n");
        cpu_matmul<half,float>(a,b,d,n,__half2float);
        printf("sum of d (CPU check) elements: %lf\n",sumMatrix<float, double>(d,n,n,convert<float,double>));
        assert(compareMatrices<float>(c,d,n,n));
        delete[] d;
    }
    #endif
    cudaEventRecord(cpu_stop);
    
    #ifdef DEBUG
    if(DEBUG>2){
        printMatrix<float,float>(c,n,n, convert<float,float>);
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
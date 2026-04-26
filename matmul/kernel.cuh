#include "funcs.h"

template<typename T, int tile_size>
__global__ void matmulshared(T* a, T* b, T* c, int n){
    __shared__ T shared_a[tile_size][tile_size];
    __shared__ T shared_b[tile_size][tile_size];

    int row=blockIdx.y*tile_size+threadIdx.y;
    int col=blockIdx.x*tile_size+threadIdx.x;
    T sum=0;

    int num_tiles=(n+tile_size-1)/tile_size;

    for(int k=0; k<num_tiles; k++){
        shared_a[threadIdx.y][threadIdx.x]=a[row*n+(k*tile_size+threadIdx.x)];
        shared_b[threadIdx.y][threadIdx.x]=b[n*(k*tile_size+threadIdx.y)+col];
    
        __syncthreads();
    
        for(int k=0; k<tile_size; k++){
            sum+=shared_a[threadIdx.y][k]*shared_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    c[row*n+col]=sum;
}

template<typename T>
__global__ void matmulnaive(T* a, T* b, T* c, int n){

    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    T sum=0;
    
    for(int k=0; k<n; k++){
        sum+=a[row*n+k]*b[k*n+col];
    }
    c[row*n+col]=sum;
}
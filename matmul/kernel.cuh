#include "funcs.h"

template<typename T, typename M, int tile_size>
__global__ void matmulshared(T* a, T* b, T* c, int n){
    __shared__ T shared_a[tile_size][tile_size];
    __shared__ T shared_b[tile_size][tile_size];

    int row=blockIdx.y*tile_size+threadIdx.y;
    int col=blockIdx.x*tile_size+threadIdx.x*4;
    M sum={0,0,0,0};

    int num_tiles=(n+tile_size-1)/tile_size;

    for(int k=0; k<num_tiles; k++){
        *reinterpret_cast<M*>(&shared_a[threadIdx.y][threadIdx.x*4])=*reinterpret_cast<M*>(&a[row*n+(k*tile_size+threadIdx.x*4)]);
        *reinterpret_cast<M*>(&shared_b[threadIdx.y][threadIdx.x*4])=*reinterpret_cast<M*>(&b[n*(k*tile_size+threadIdx.y)+col]);
    
        __syncthreads();
    
        for(int t=0; t<tile_size; t++){
            T a_val=shared_a[threadIdx.y][t];
            M b_val=*reinterpret_cast<M*>(&shared_b[t][threadIdx.x*4]);
            sum.x+=a_val * b_val.x;
            sum.y+=a_val * b_val.y;
            sum.z+=a_val * b_val.z;
            sum.w+=a_val * b_val.w;
        }
        __syncthreads();
    }
    *reinterpret_cast<M*>(&c[row*n+col])=sum;
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
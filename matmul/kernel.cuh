#include "funcs.h"

using namespace nvcuda::wmma;

template <typename T, typename O, int tile_size>
__global__ void matmultensor(T* a, T* b, O* c, int n){
    int warps_per_block=blockDim.x/32;
    int warpid=int(threadIdx.x/32)+int(blockDim.x*blockIdx.x/32);

    int tiles_per_dim=n/tile_size;
    int numtiles=tiles_per_dim*tiles_per_dim;

    if(warpid>=numtiles) return;

    int tile_row=warpid/tiles_per_dim;
    int tile_col=warpid % tiles_per_dim;

    fragment<matrix_a, tile_size,tile_size,tile_size,T,row_major> a_fragment;
    fragment<matrix_b, tile_size,tile_size,tile_size,T,row_major> b_fragment;
    fragment<accumulator, tile_size,tile_size,tile_size,O> c_fragment;

    fill_fragment(c_fragment,0);

    for(int k=0; k<n; k+=tile_size){
        T* a_block=&a[tile_row*tile_size*n + k];
        T* b_block=&b[tile_col*tile_size+n*k];

        load_matrix_sync(a_fragment,a_block,n);
        load_matrix_sync(b_fragment,b_block,n);

        mma_sync(c_fragment,a_fragment,b_fragment,c_fragment);
    }

    O* c_tile=&c[tile_row*tile_size*n + tile_col*tile_size];
    store_matrix_sync(c_tile,c_fragment,n,mem_row_major);
}

template<typename T, typename M, int tile_size>
__global__ void matmulvector(T* a, T* b, T* c, int n){
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
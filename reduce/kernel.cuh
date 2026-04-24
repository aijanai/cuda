#include "funcs.h"


template<typename T>
__device__ void unroll(volatile T* data, int tid){
    data[tid]+=data[tid+32];
    data[tid]+=data[tid+16];
    data[tid]+=data[tid+8];
    data[tid]+=data[tid+4];
    data[tid]+=data[tid+2];
    data[tid]+=data[tid+1];
}

template<typename T, unsigned int blocksize>
__global__ void reduce(T* input, T n){
    __shared__ T shared[blocksize];
    int tid=threadIdx.x;
    int i=tid+blockIdx.x*2*blockDim.x;
    // initialize warp local sum
    T sum=0;

    // populate local memory from central
    #ifdef DEBUG
    if(DEBUG>2){
        printf(fmtKernelPopulateSharedMsg<T>(),tid, blockIdx.x, tid, i, input[i]);
    }
    #endif
    sum=(i<n) ? input[i] : 0;

    if(i+blockDim.x<n){
        #ifdef DEBUG
        if(DEBUG>2){
            printf(fmtKernelInProgressMsg<T>(),tid,blockIdx.x, blockDim.x, tid, i+blockDim.x, sum, input[i+blockDim.x] );
        }
        #endif
        sum+=input[i+blockDim.x];
    }else{
        #ifdef DEBUG
        if(DEBUG>2){
            printf(fmtKernelInProgressSkipMsg<T>(),tid,blockIdx.x, blockDim.x, tid, i+blockDim.x );
        }
        #endif

    }
    __syncthreads();

    #ifdef DEBUG
    if(DEBUG>2){
        if(tid==0){
            printf("\n");
        }
        __syncthreads();
    }
    #endif


    // at this stage 512 elements have been summed to first 256 threads
    // now, for each warp (32 threads), sum together members of warp
    for(int stride=warpSize>>1; stride >0; stride>>=1){
        sum+=__shfl_down_sync(0xffffffff,sum,stride);
    }

    // now all values are scattered in an array at position multiples of warpsize, because the reduction happened within warps
    // the positions are within [0,... block/warpsize], so for 256 threads these have all been moved to [0..8]
    if(tid % warpSize ==0){
        shared[tid/warpSize]=sum;
    }
    __syncthreads();
    
    // reduce leftovers [0,... block/warpsize]
    if(tid<warpSize){
        sum=(tid<(blockDim.x/warpSize))? shared[tid]: 0;
        for(int stride=warpSize>>1; stride >0; stride>>=1){
            sum+=__shfl_down_sync(0xffffffff,sum,stride);
        }
    }
    
    // write back to central memory the reduced value
    if(tid ==0){
        #ifdef DEBUG
        if(DEBUG>2){
            printf(fmtKernelDefragFinalMsg<T>(),tid, blockIdx.x, blockIdx.x, 0, shared[0]);
        }
        #endif
        input[blockIdx.x]=sum;
    }
}
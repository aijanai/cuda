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

    // populate shared memory
    #ifdef DEBUG
    if(DEBUG>2){
        printf(fmtKernelPopulateSharedMsg<T>(),tid, blockIdx.x, tid, i, input[i]);
    }
    #endif
    shared[tid]= (i<n) ? input[i] : 0;

    if(i+blockDim.x<n){
        #ifdef DEBUG
        if(DEBUG>2){
            printf(fmtKernelInProgressMsg<T>(),tid,blockIdx.x, blockDim.x, tid, i+blockDim.x, shared[tid], input[i+blockDim.x] );
        }
        #endif
        shared[tid]+=input[i+blockDim.x];
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

    // reduce iteratively within same block
    for(int stride=blockDim.x>>1; stride>32; stride>>=1){
        if(tid<stride){
            #ifdef DEBUG
            if(DEBUG>2){
                printf(fmtKernelInProgressMsg<T>(),tid,blockIdx.x, stride, tid, tid+stride, shared[tid], shared[tid+stride] );
            }
            #endif
            shared[tid]+=shared[tid+stride];
        }
        __syncthreads();
    }

    // shortcut for last iterations
    if(tid<32){
        unroll(shared,tid);
    }

    // write back to central memory the reduced value
    if(tid ==0){
        #ifdef DEBUG
        if(DEBUG>2){
            printf(fmtKernelDefragFinalMsg<T>(),tid, blockIdx.x, blockIdx.x, 0, shared[0]);
        }
        #endif
        input[blockIdx.x]=shared[0];
    }
}
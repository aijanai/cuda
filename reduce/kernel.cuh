#include "funcs.h"

template<typename T, unsigned int blocksize>
__global__ void reduce(T* input, T n){
    //__shared__ T shared[blocksize];
    int tid=threadIdx.x;
    int i=tid+blockIdx.x*blockDim.x;
    //shared[tid]= (i<n) ? input[i] : 0;

    for(int stride=1; stride<blockDim.x; stride*=2){
        if(i+stride<n && (tid % (2*stride)) == 0){
            #ifdef DEBUG
            if(DEBUG>1){
                printf(fmtKernelInProgressMsg<T>(),tid,blockIdx.x, stride, i, i+stride, input[i], input[i+stride] );
            }
            #endif
            input[i]+=input[i+stride];
            //shared[tid]+=shared[tid+stride];
        }
        __syncthreads();
    }

    if(tid ==0){
        #ifdef DEBUG
        if(DEBUG>1){
            printf(fmtKernelDefragMsg<T>(),tid, blockIdx.x, blockIdx.x, i, input[i]);
        }
        #endif
        input[blockIdx.x]=input[i];
        //input[blockIdx.x]=shared[0];
    }
}
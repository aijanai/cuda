#include "funcs.h"

template<typename T>
__global__ void reduce(T* input, T n){
    int tid=threadIdx.x;
    int i=tid+blockIdx.x*blockDim.x;
    int stride;
    int levels = static_cast<int>(log2f(blockDim.x));
    for(int l=0; l<levels; l++){
        stride=static_cast<int>(pow(2,l));
        if(i+stride<n && (tid % (2*stride)) == 0){
            #ifdef DEBUG
            if(DEBUG>1){
                printf(fmtKernelInProgressMsg<T>(),tid,l,blockIdx.x, stride, i, i+stride, input[i], input[i+stride] );
            }
            #endif
            input[i]+=input[i+stride];
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
    }
}
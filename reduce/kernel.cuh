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
            printf("tid %d lev %d block %d stride %d (%d-th val + %d-th val): (%d + %d)\n",tid,l,blockIdx.x, stride, i, i+stride, input[i], input[i+stride] );
            #endif
            input[i]+=input[i+stride];
        }
        __syncthreads();
    }

    if(tid ==0){
        #ifdef DEBUG
        printf("tid %d block %d (%d-th val <- %d-th): %d\n",tid, blockIdx.x, blockIdx.x, i, input[i]);
        #endif
        input[blockIdx.x]=input[i];
    }
}
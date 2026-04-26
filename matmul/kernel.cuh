#include "funcs.h"

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
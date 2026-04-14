#include "funcs.h"

__global__ void add(int* a, int* b, int* c){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    add_impl(a,b,c,i);
}

__host__ __device__ void add_impl(int* a, int* b, int* c, int i){
    c[i]=a[i]+b[i];
}
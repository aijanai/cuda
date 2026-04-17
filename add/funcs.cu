#include "funcs.h"

__global__ void add(int* a, int* b, int* c, int n){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n){
        add_impl(a,b,c,i);
    }
}

__host__ __device__ void add_impl(int* a, int* b, int* c, int i){
    c[i]=a[i]+b[i];
}

__global__ void add2d(int* a, int* b, int* c, int n, int m){
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    if(x<n and y<m){
        int i=x+m*y;
        add_impl(a,b,c,i);
    }
}

void printMatrix(int* a, int n, int m){
    int k;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            k=j+n*i;
            printf("%d ",a[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void printArray(int* a, int n){
    for(int i=0; i<n; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

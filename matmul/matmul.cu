#include <cuda.h>
#include <cassert>
#include "funcs.h"
#include "kernel.cuh"


int main(int argc, char** argv){

    if(argc<1){
        printf("Usage: %s <len>\n", argv[0]);
        return -2;
    }

    int n=atoi(argv[1]);
    auto* a = initializeMatrix<int>(n,n);
    auto* b = initializeMatrix<int>(n,n);
    auto* c = initializeMatrix<int>(n,n);

    int *ga, *gb, *gc;

    size_t SIZE=getByteSize<int>(n,n);

    cudaMalloc((void**) &ga, SIZE);
    cudaMalloc((void**) &gb, SIZE);
    cudaMalloc((void**) &gc, SIZE);

    for(int i=0; i<n*n; i++){
        a[i]=1;
        b[i]=a[i];
        c[i]=0;
    }

    //b=initializeIdentity<int>(n);

    #ifdef DEBUG
    if(DEBUG>1){
        printf("a:\n");
        printMatrix(a,n,n);
        printf("b:\n");
        printMatrix(b,n,n);
    }
    #endif

    cudaMemcpy(ga,a,SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(gb,b,SIZE,cudaMemcpyHostToDevice);

    //matmulnaive<<<numblocks,blocksize>>>(ga,gb,gc,n);



    cudaMemcpy(c,gc,SIZE,cudaMemcpyDeviceToHost);

    cpu_matmul(a,b,c,n);

    printMatrix(c,n,n);

    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);

    cudaDeviceReset();
    return 0;
}
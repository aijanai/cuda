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
    auto* d = initializeMatrix<int>(n,n);

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
    if(DEBUG>2){
        printf("a:\n");
        printMatrix(a,n,n);
        printf("b:\n");
        printMatrix(b,n,n);
    }
    #endif

    cudaMemcpy(ga,a,SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(gb,b,SIZE,cudaMemcpyHostToDevice);

    int tile_size=16;
    dim3 blocksize(tile_size,tile_size);
    dim3 grid_size((n+tile_size-1)/tile_size, (n+tile_size-1)/tile_size);

    matmulnaive<<<grid_size,blocksize>>>(ga,gb,gc,n);

    cudaMemcpy(c,gc,SIZE,cudaMemcpyDeviceToHost);

    cpu_matmul(a,b,d,n);

    printf("sum of a elements: %d\n",sumMatrix<int>(a,n,n));
    printf("sum of b elements: %d\n",sumMatrix<int>(b,n,n));
    printf("sum of c elements: %d\n",sumMatrix<int>(c,n,n));
    printf("sum of d elements: %d\n",sumMatrix<int>(d,n,n));
    assert(compareMatricesLinear<int>(c,d,n*n));

    #ifdef DEBUG
    if(DEBUG>2){
        printMatrix(c,n,n);
    }
    #endif

    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);

    cudaDeviceReset();
    return 0;
}
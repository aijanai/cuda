#ifndef FUNCS_H
#define FUNCS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

template <typename T> __device__ __host__ const char* fmt();
template <> inline __device__ __host__ const char* fmt<int>()           { return "%d "; }
template <> inline __device__ __host__ const char* fmt<float>()         { return "%f "; }
template <> inline __device__ __host__ const char* fmt<double>()        { return "%lf "; }
template <> inline __device__ __host__ const char* fmt<unsigned long>() { return "%lu "; }

template <typename T> __device__ __host__ const char* fmtKernelInProgressMsg();
template <> inline __device__ __host__ const char* fmtKernelInProgressMsg<int>() {
    return  "tid %d lev %d block %d stride %d (%d-th val + %d-th val): (%d + %d)\n";
}
template <> inline __device__ __host__ const char* fmtKernelInProgressMsg<unsigned long>() {
    return  "tid %d lev %d block %d stride %d (%lu-th val + %lu-th val): (%lu + %lu)\n";
}
template <typename T> __device__ __host__ const char* fmtKernelDefragMsg();
template <> inline __device__ __host__ const char* fmtKernelDefragMsg<int>() {
    return "tid %d block %d (%d-th val <- %d-th): %d\n";
}
template <> inline __device__ __host__ const char* fmtKernelDefragMsg<unsigned long>() {
    return "tid %d block %d (%lu-th val <- %lu-th): %lu\n";
}

template<typename T>
__host__ __device__ void printArray(T* a, T n){
    for(T i=0; i<n; i++){
        printf(fmt<T>(), a[i]);
    }
    printf("\n");
}

template<typename T>
__host__ __device__ T sumArrayLinear(T* a, T n){
    T sum=0;
    for(T i=0; i<n; i++){
        sum+=a[i];
    }
    return sum;
}

#endif
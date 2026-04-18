#ifndef FUNCS_H
#define FUNCS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

template<typename T>
__host__ __device__ void printArray(T* a, T n){
    for(T i=0; i<n; i++){
        printf("%d ", a[i]);
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

template <typename T> __device__ __host__ const char* fmt();
template <> inline __device__ __host__ const char* fmt<int>()          { return "%d"; }
template <> inline __device__ __host__ const char* fmt<float>()        { return "%f"; }
template <> inline __device__ __host__ const char* fmt<double>()       { return "%lf"; }
template <> inline __device__ __host__ const char* fmt<unsigned long>() { return "%lu"; }

#endif
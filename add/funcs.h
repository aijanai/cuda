#ifndef FUNCS_H
#define FUNCS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void add(int* a, int* b, int* c, int n);
__host__ __device__ void add_impl(int* a, int* b, int* c, int i);

__global__ void add2d(int* a, int* b, int* c, int n, int m);

void printArray(int* a, int n);
void printMatrix(int* a, int n, int m);
#endif
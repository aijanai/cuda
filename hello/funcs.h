#ifndef FUNCS_H
#define FUNCS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>


__global__ void add(int* a, int* b, int* c);

__host__ __device__ void add_impl(int* a, int* b, int* c, int i);

#endif
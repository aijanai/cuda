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

/*
template <typename T> __device__ __host__ const char* fmtKernelInProgressMsg();
template <> inline __device__ __host__ const char* fmtKernelInProgressMsg<int>() {
    return  "tid %d block %d stride %d (shared %d-th val + input %d-th val): (%d + %d)\n";
}
template <> inline __device__ __host__ const char* fmtKernelInProgressMsg<unsigned long>() {
    return  "tid %d block %d stride %d (shared %d-th val + input %d-th val): (%lu + %lu)\n";
}
template <> inline __device__ __host__ const char* fmtKernelInProgressMsg<float>() {
    return  "tid %d block %d stride %d (shared %d-th val + input %d-th val): (%f + %f)\n";
}

template <typename T> __device__ __host__ const char* fmtKernelInProgressSkipMsg();
template <> inline __device__ __host__ const char* fmtKernelInProgressSkipMsg<int>() {
    return  "skip tid %d block %d stride %d (%d-th val + %d-th val)\n";
}
template <> inline __device__ __host__ const char* fmtKernelInProgressSkipMsg<unsigned long>() {
    return  "skip tid %d block %d stride %d (%d-th val + %d-th val)\n";
}
template <> inline __device__ __host__ const char* fmtKernelInProgressSkipMsg<float>() {
    return  "skip tid %d block %d stride %d (%d-th val + %d-th val)\n";
}

template <typename T> __device__ __host__ const char* fmtKernelDefragFinalMsg();
template <> inline __device__ __host__ const char* fmtKernelDefragFinalMsg<int>() {
    return "tid %d block %d (input %d-th val <- shared %d-th): %d\n";
}
template <> inline __device__ __host__ const char* fmtKernelDefragFinalMsg<unsigned long>() {
    return "tid %d block %d (input %d-th val <- shared %d-th): %lu\n";
}
template <> inline __device__ __host__ const char* fmtKernelDefragFinalMsg<float>() {
    return "tid %d block %d (input %d-th val <- shared %d-th): %f\n";
}

template <typename T> __device__ __host__ const char* fmtKernelPopulateSharedMsg();
template <> inline __device__ __host__ const char* fmtKernelPopulateSharedMsg<int>() {
    return "tid %d block %d (shared %d-th val <- input %d-th): %d\n";
}
template <> inline __device__ __host__ const char* fmtKernelPopulateSharedMsg<unsigned long>() {
    return "tid %d block %d (shared %d-th val <- input %d-th): %lu\n";
}
template <> inline __device__ __host__ const char* fmtKernelPopulateSharedMsg<float>() {
    return "tid %d block %d (shared %d-th val <- input %d-th): %f\n";
}
*/

/*
template <typename T> __device__ __host__ const char* fmtExpected();
template <> inline __device__ __host__ const char* fmtExpected<unsigned long>() {
    return "Expected sum: %lu, got %lu\n";
}
template <> inline __device__ __host__ const char* fmtExpected<int>() {
    return "Expected sum: %d, got %d\n";
}
template <> inline __device__ __host__ const char* fmtExpected<float>() {
    return "Expected sum: %f, got %f\n";
}
template <typename T> __device__ __host__ const char* fmtRes();
template <> inline __device__ __host__ const char* fmtRes<unsigned long>() {
    return "\nRESULT: %lu\n\n";
}
template <> inline __device__ __host__ const char* fmtRes<int>() {
    return "\nRESULT: %d\n\n";
}
template <> inline __device__ __host__ const char* fmtRes<float>() {
    return "\nRESULT: %f\n\n";
}
template<typename T>
__host__ void checkAndDisplay(T* a, T* b, unsigned long n){
    T sum=sumArrayLinear<T>(b,n);
    if(sum!=a[0]){
        printf(fmtExpected<T>(), sum, a[0]);
        assert(sum==a[0]);
    }
    printf(fmtRes<T>(), a[0]);
}
*/

/*
template<typename T>
void dumpFirstN(T* a, unsigned long reduced_n){
    printf("dump first %lu records: \n",reduced_n);
    for(int i=0; i<reduced_n; i++){
        printf(fmt<T>(),a[i]);
    }
    printf("\n");
}
*/


template<typename T>
size_t getByteSize(int n, int m){
    return n*m*sizeof(T);
}

template<typename T>
T* initializeMatrix(int n, int m) {
    T *a;

    // alloc on CPU
    a = new T[n*m];
    return a;
}

template<typename T>
__host__ __device__ void printMatrix(T* a, int n, int m){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            printf(fmt<T>(), a[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
}

template<typename T, typename X>
__host__ __device__ X sumMatrix(T* a, int n, int m){
    X sum=0;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            sum+=a[i*n+j];
        }
    }
    return sum;
}

template<typename T>
__host__ __device__ void cpu_matmul(T* a, T* b, T* c, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            T sum=0;
            for(int k=0; k<n; k++){
                sum+=a[i*n+k]*b[k*n+j];
            }
            c[i*n+j]=sum;
        }
    }

}

template<typename T>
__host__ __device__ bool compareMatrices(T* a, T* b, int n, int m){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if (a[i*n+j] != b[i*n+j]){
                return false;
            }
        }
    }
    return true;
}

template<typename T>
__host__ __device__ bool compareMatricesLinear(T* a, T* b, int n){
        for(int i=0; i<n; i++){
            if (a[i] != b[i]){
                return false;
            }
        }
        return true;
}


template<typename T>
T* initializeIdentity(int n) {
    T *a;

    // alloc on CPU
    a = new T[n*n];

    T val;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i==j){
                val=1;
            }else{
                val=0;
            }
            a[i*n+j]=val;
        }
    }

    return a;
}

/*
template<typename T>
T* initializeCUDAArray() {
    T *ga=nullptr;
    return ga;
}
*/


/*
template <typename T> __device__ __host__ const char* fmtPartials();
template <> inline __device__ __host__ const char* fmtPartials<unsigned long>() {
    return "sum of first %d nums: %lu\n";
}
template <> inline __device__ __host__ const char* fmtPartials<int>() {
    return "sum of first %d nums: %d\n";
}
template <> inline __device__ __host__ const char* fmtPartials<float>() {
    return "sum of first %d nums: %f\n";
}
*/
/*
template<typename T>
int printPartials(T* ga, unsigned long n, unsigned long reduced_n, int debug_lvl, bool first_run){
    auto c=initializeArray<T>(reduced_n);
    // copy partials
    cudaError_t err=cudaMemcpy(c, ga, reduced_n*sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    if(debug_lvl>1 && !first_run){
        printf("dump partial sums in intermediate array:\n");
        printArray<T>(c,reduced_n);
    }
    
    if(debug_lvl>0){
        printf(fmtPartials<T>(),reduced_n,sumArrayLinear<T>(c,reduced_n));
    }
    
    delete[] c;
    return 0;
}


template<typename T>
int readBackResults(T* a, T* ga){
    cudaError_t err=cudaMemcpy(a, ga, 1*sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("CUDA error: gc memcpy %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
*/
#endif
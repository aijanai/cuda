# cuda

A collection of CUDA C++ examples exploring GPU programming concepts, from basic thread identification to performance profiling.

## Structure

```
cuda/
├── hello/        # Basic kernel: block, thread, and warp identification
├── add/          # Vector addition with GPU/CPU comparison and profiling
└── properties/   # Query and print GPU device properties
```

## Examples

### hello

Launches a simple kernel across 2 blocks of 64 threads each. Each thread prints its block ID, thread ID, and warp ID.

```bash
cd hello && make && ./hello
```

### add

Vector addition benchmark comparing GPU and CPU performance. The GPU version processes 10 billion elements in 256 MB chunks and profiles each stage (malloc, host→device copy, kernel, device→host copy) using CUDA events.

The implementation is split into:
- `funcs.cu` / `funcs.h` — shared kernel (`__global__`) and implementation (`__host__ __device__`) so the same logic runs on both CPU and GPU
- `add.cu` — GPU driver
- `cpu_add.cu` — CPU-only driver for benchmarking

```bash
cd add && make
./add        # GPU version
./cpu_add    # CPU version
```

### properties

Queries all available CUDA devices and prints hardware details: name, streaming multiprocessor count, max blocks/threads, shared memory, total global memory, and compute capability.

```bash
cd properties && make && ./properties
```

## Build

All targets use `nvcc` with native architecture detection and `clang++` as the host compiler:

```bash
make   # in any subdirectory
```

Compiled binaries and intermediate files are excluded via `.gitignore`.

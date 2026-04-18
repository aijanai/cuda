CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall -g -DDEBUG=1

TARGETS= reduce

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS}

reduce: reduce.cu funcs.h kernel.cuh
	$(CC) $(OPTS) $< -o $@
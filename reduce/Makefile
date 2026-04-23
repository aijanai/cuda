CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall #-DDEBUG=3 #-g -G

TARGETS= reduce

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS}

reduce: reduce.cu funcs.h kernel.cuh
	$(CC) $(OPTS) $< -o $@
CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall #-DDEBUG=1 #-g -G

TARGETS= reduce

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS} *preprocessed*

reduce: reduce.cu funcs.h kernel.cuh
	$(CC) $(OPTS) $< -o $@

kernel-preprocessed.cu: kernel.cuh
	cp kernel.cuh kernel.cu
	$(CC) -E kernel.cu -o $@
	rm kernel.cu
	grep -v ^# $@ > $@2
	mv $@2 $@
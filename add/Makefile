CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall -g #-DDEBUG=1

TARGETS= add cpu_add chunked_add static_add add2d cpu_add2d

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS}

static_add: static_add.cu
	$(CC) $(OPTS) $< -o $@

add: add.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@

add2d: add2d.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@

cpu_add2d: cpu_add2d.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@

chunked_add: chunked_add.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@

funcs.o: funcs.cu funcs.h
	$(CC) $(OPTS) -c $< -o $@

cpu_add: cpu_add.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@
CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall -g #-DEBUG=1

TARGETS= add cpu_add properties

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS}

add: add.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@

funcs.o: funcs.cu funcs.h
	$(CC) $(OPTS) -c $< -o $@

cpu_add: cpu_add.cu funcs.o funcs.h
	$(CC) $(OPTS) $< $(word 2, $^) -o $@
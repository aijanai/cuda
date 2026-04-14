CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall -g #-DEBUG=1

TARGETS= hello

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS}

hello: hello.cu
	$(CC) $(OPTS) $< -o $@
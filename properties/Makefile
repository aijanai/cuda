CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall -g #-DEBUG=1

TARGETS= properties

all: ${TARGETS}

clean:
	rm -f *.o *asm main ${TARGETS}

properties: properties.cu
	$(CC) $(OPTS) $< -o $@

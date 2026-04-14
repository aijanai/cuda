CC=nvcc
OPTS=-arch=native -ccbin=clang++ -Xcompiler -Wall -g #-O0

all: hello add

clean:
	rm -f *.o hello *asm main add

hello: hello.cu
	$(CC) $(OPTS) hello.cu -o hello

add: add.cu
	$(CC) $(OPTS) add.cu -o add
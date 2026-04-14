
CC=nvcc
OPTS=-arch=native #-Wall -std=c++17 -g -O0

all: main

clean:
	rm -f *.o hello *asm main

main: kernel.cu
	$(CC) $(OPTS) kernel.cu -o main
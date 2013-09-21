INCLUDE_PATH = /usr/local/cuda/include
LIB_PATH = /usr/local/cuda/lib

CUDA_CC = /usr/local/cuda/bin/nvcc
CUDA_CFLAGS = -I$(INCLUDE_PATH)

CC = g++
CFLAGS = -I$(INCLUDE_PATH) -L$(LIB_PATH) -lcudart

main: ga.o
	$(CC) -w main.c ga.o -o main.out $(CFLAGS)

ga.o:
	$(CUDA_CC) -w genetic_algorithm.cu -c -o ga.o $(CUDA_CFLAGS)

clean:
	rm -rf *.o main.out
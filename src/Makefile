CUDA_PATH = /usr/local/cuda-5.5
INCLUDE_PATH = $(CUDA_PATH)/include
LIB_PATH = $(CUDA_PATH)/lib64

CUDA_CC = $(CUDA_PATH)/bin/nvcc
CUDA_CFLAGS = -I$(INCLUDE_PATH)

CC = g++
CFLAGS = -I$(INCLUDE_PATH) -L$(LIB_PATH) -lcudart

main: ga.o
	$(CC) main.c ga.o -o main.out $(CFLAGS)

ga.o:
	$(CUDA_CC) genetic_algorithm.cu -c -o ga.o $(CUDA_CFLAGS)

clean:
	rm -rf *.o *.out
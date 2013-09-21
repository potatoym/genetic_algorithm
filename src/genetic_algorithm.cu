#include <cuda.h>
#include <stdlib.h>
#include <time.h>

#include <stdio.h>

#include "genetic_algorithm.h"

template <class T> void swap (T& a, T& b){
	T c(a); a = b; b = c;
}

__device__ int rand_int(int *seed){
	unsigned int ret = 0;
	unsigned int xi = *(unsigned int *)seed;
	unsigned int m = 65537 * 67777;

	xi = (xi * xi) % m;
	*seed = *(unsigned int *)&xi;

	return xi % RAND_MAX_GA;
}

__device__ float rand_float(int *seed){
	float r = (float)rand_int(seed);
	return r / (float)RAND_MAX_GA;
}

__device__ void generateSpecimen(specimen *s, int *random_seed, size_t index){
	int i;
	for(i = 0; i < specimenbits; ++i)
		s[0].c[i] = (int) rand_int(random_seed) % 2;
}

__device__ int selectSpecimen(specimen *pop, int size, int *random_seed){
	int i, j;
	i = rand_int(random_seed) % size;
	j = (rand_int(random_seed) % (size - 1) + i + 1) % size;

	return (pop[i].fitness > pop[j].fitness) ? i : j;
}

__device__ void crossover(specimen *parent, specimen *offspring, int *random_seed){
	int i;
	int cpoint = rand_int(random_seed) % specimenbits;
	for(i = 0; i < specimenbits; ++i){
		int part = (i < cpoint) ? 1 : 0;
		offspring[0].c[i] = parent[part].c[i];
		offspring[1].c[i] = parent[1-part].c[i];
	}

	offspring[0].fitness = 0;
	offspring[1].fitness = 0;
}

__device__ void mutate(specimen *parent, int *random_seed){
	int i;
	for(i = 0; i < specimenbits; ++i){
		if(rand_float(random_seed) < pmutation){
			parent->c[i] = 1 - parent->c[i];
		}
	}
}

__device__ float fitness(const specimen *sp){
	int s = 0, i = 0;
	for(i = 0; i < specimenbits; ++i){
		s += (int)(sp->c[i]);
	}

	return ((float)s) / (float)specimenbits;
}

__global__ void initPopulation(specimen *pop, const int size, const int random_seed){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < size){
		int seed = random_seed + i, j;
		for(j = 0; j < specimenbits; ++j)
			pop[i].c[j] = rand_int(&seed) % 2;
	}
}

__global__ void newGeneration(specimen *pop, specimen *newpop, const int size, const int random_seed){
	const int i = 2 * (blockIdx.x*blockDim.x + threadIdx.x);
	if(i >= size) return;

	specimen parent[2], offspring[2];
	int seed = random_seed + i;

	parent[0] = pop[selectSpecimen(pop, size, &seed)];
	parent[1] = pop[selectSpecimen(pop, size, &seed)];

	if(rand_float(&seed) < pcross){
		crossover(parent, offspring, &seed);
	} else {
		offspring[0] = parent[0];
		offspring[1] = parent[1];
	}

	mutate(&offspring[0], &seed);
	mutate(&offspring[1], &seed);
	newpop[i] = offspring[0];
	newpop[i+1] = offspring[1];
}

__global__ void countFitness(specimen *pop, const int size){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < size){
		const specimen sp = pop[i];
		pop[i].fitness = fitness(&sp);
	}
}

__global__ void findBestSpecimen(specimen *pop, const int size){
	const int index = threadIdx.x;
	int bestIndex = index, i;

	for(i = index+THREADS; i < size; i += THREADS){
		if(pop[bestIndex].fitness < pop[i].fitness)
			bestIndex = i;
	}

	__shared__ int buffer[THREADS];
	buffer[index] = bestIndex;
	__syncthreads();

	if(index == 0){
		for(i = 0; i < THREADS; ++i)
			if(pop[bestIndex].fitness < pop[i].fitness)
				bestIndex = i;

		pop[0] = pop[bestIndex];
	}
}

void genetic_algorithm(){
	srand (time(NULL));

	const int population = THREADS * BLOCKS;

	specimen *devPopulation = 0, *devNewPopulation = 0;
	cudaMalloc((void**)&devPopulation, sizeof(specimen) * population);
	cudaMalloc((void**)&devNewPopulation, sizeof(specimen) * population);

	initPopulation<<<BLOCKS, THREADS>>>(devPopulation, population, rand() % RAND_MAX_GA);
	cudaThreadSynchronize();

	int i;
	for(i = 0; i < 1000; ++i){

		countFitness<<<BLOCKS, THREADS>>>(devPopulation, population);
		newGeneration<<<BLOCKS, THREADS>>>(devPopulation, devNewPopulation, population, rand() % RAND_MAX_GA);
		cudaThreadSynchronize();
		swap(devPopulation, devNewPopulation); 
	}

	findBestSpecimen<<<1, THREADS>>>(devPopulation, population);
	cudaThreadSynchronize();

	specimen best;
	cudaMemcpy(&best, &devPopulation[0], sizeof(specimen), cudaMemcpyDeviceToHost);

	printf("%f\n", best.fitness);

	cudaFree(devPopulation);
	cudaFree(devNewPopulation);
}
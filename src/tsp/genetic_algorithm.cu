#include <cuda.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <time.h>

#include <stdio.h>

#include "genetic_algorithm.h"

__device__ __constant__ int cudaCity[specimenbits];

template <class T> __device__ __host__ void swap (T& a, T& b){
	T c(a); a = b; b = c;
}

__global__ void setupStates(curandState *states, int length, unsigned long long seed, unsigned long long offset){
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < length){
		curand_init(seed, id, offset + id, &states[id]);
	}
}

__global__ void initPopulation(specimen *pop, const int size, curandState *states){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id < size){
		curandState localState = states[id];
		int j, i;
		for(j = 0; j < specimenbits; ++j)
			pop[id].c[j] = cudaCity[j];

		for(j = specimenbits - 1; j >= 0; --j){
			i = curand(&localState) % specimenbits;
			if(i != j){ swap(pop[id].c[j], pop[id].c[i]); }
		}
		states[i] = localState;
	}
}

__device__ int fitness(const specimen *sp, int* cities){
	int s = 0, i;
	for(i = 0; i < specimenbits-1; ++i)
		s += cities[(sp->c[i]*specimenbits) + sp->c[i+1]];
	s += cities[(sp->c[i]*specimenbits) + sp->c[0]];

	return s;
}

__global__ void countFitness(specimen *pop, int *cities, int length){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < length){
		const specimen sp = pop[i];
		pop[i].fitness = fitness(&sp, cities);
	}
}

__device__ int selectSpecimen(specimen *pop, int size, curandState &localState){
	int i, j;
	i = curand(&localState) % size;
	j = (curand(&localState) % (size - 1) + i + 1) % size;

	return (pop[i].fitness < pop[j].fitness) ? i : j;
}

/**
	читай описание алгоритма в книге description genetic algorithm на стр 220.
*/
__device__ void setMaska(specimen &parent, specimen &offspring, char *keys){
	int i, j = 0, k;
	for(i = 0; i < specimenbits; ++i){
		k = parent.c[i];
		if(keys[k] == 1){
			while(offspring.c[j] != -1) {j++;}
			offspring.c[j] = k;
			keys[k] = 0;
		}
	}
}

__device__ void crossover(specimen *parent, specimen *offspring, curandState &localState){
	char keys[2][specimenbits]; 
	int i, k;
	//! Инициализация
	for(i = 0; i < specimenbits; ++i){
		keys[0][i] = keys[1][i] = 1;
		offspring[0].c[i] = offspring[1].c[i] = -1;
	}
	
	//! Создаем маску и оставляем на своих местах элементы соотв. нулевым элементам.
	for(i = 0; i < specimenbits; ++i){
		if(curand_normal(&localState) > pcross_maska){
			k = parent[0].c[i];
			keys[0][k] = 0;
			offspring[0].c[i] = k;

			k = parent[1].c[i];
			keys[1][k] = 0;
			offspring[1].c[i] = k;
		}
	}

	setMaska(parent[1], offspring[0], keys[0]);
	setMaska(parent[0], offspring[1], keys[1]);

	offspring[0].fitness = 0;
	offspring[1].fitness = 0;
}

__device__ void mutate(specimen *parent, curandState &localState){
	int i, j;
	i = curand(&localState) % specimenbits;
	j = (curand(&localState) % (specimenbits - 1) + i + 1) % specimenbits;

	swap(parent->c[i], parent->c[j]);
}

__global__ void newGeneration(specimen *pop, specimen *newpop, const int size, curandState *states){
	const int i = 2 * (blockIdx.x*blockDim.x + threadIdx.x);
	if((i + 1) >= size) return;

	specimen parent[2], offspring[2];
	curandState localState = states[i];

	parent[0] = pop[selectSpecimen(pop, size, localState)];
	parent[1] = pop[selectSpecimen(pop, size, localState)];

	if(curand_normal(&localState) < pcross){
		crossover(parent, offspring, localState);
	} else {
		offspring[0] = parent[0];
		offspring[1] = parent[1];
	}

	mutate(&offspring[0], localState);
	mutate(&offspring[1], localState);
	newpop[i] = offspring[0];
	newpop[i+1] = offspring[1];
}


void print(specimen *gpu_array, int length){

	specimen *array = (specimen *)malloc(sizeof(specimen) * length);
	cudaMemcpy(array, gpu_array, sizeof(specimen) * length, cudaMemcpyDeviceToHost);

	int i, j;
	for(i = 0; i < length; ++i){
		for(j = 0; j < specimenbits; ++j)
			printf("%d ", array[i].c[j]);
		printf(" FITNESS = %d\n", array[i].fitness);
		printf("\n");
	}

	free(array);
}

void printCities(int *array){
	int i,j;
	printf("\n");
	for(i = 0; i < specimenbits; ++i){
		for(j = 0; j < specimenbits; ++j)
			printf("%d ", array[i*specimenbits + j]);
		printf("\n");
	}
}

__global__ void findBestSpecimen(specimen *pop, const int size){
	const int index = threadIdx.x;
	if(index >= THREADS) return;

	int bestIndex = index, i;
	for(i = index+THREADS; i < size; i += THREADS){
		if(pop[bestIndex].fitness > pop[i].fitness)
			bestIndex = i;
	}

	__shared__ int buffer[THREADS];
	buffer[index] = bestIndex;
	__syncthreads();

	if(index == 0){
		for(i = 0; i < THREADS; ++i)
			if(pop[bestIndex].fitness > pop[ buffer[i] ].fitness)
				bestIndex = buffer[i];

		pop[0] = pop[bestIndex];
	}
}

int* setupCities(){
	static int matrix[specimenbits * specimenbits];

	int i, j;
	for(i = 0; i < specimenbits; ++i){
		for(j = 0; j < specimenbits; ++j)
			matrix[i * specimenbits + j] = (i == j) ? 0 : rand() % 10 + 1;
	}

	//! Тестовый пример из книги
	// static int matrix[specimenbits * specimenbits] = {100, 4, 6, 2, 9,
	// 												  4, 100, 3, 2, 9,
	// 												  6, 3, 100, 5, 9,
	// 												  2, 2, 5, 100, 8,
	// 												  9, 9, 9, 8, 100};

	return matrix;
}

void genetic_algorithm(){
	srand (time(NULL));

	const int length = THREADS * BLOCKS;

	//! Создание матрицы городов
	int *cities = setupCities();

	int *cudaCities = 0;
	cudaMalloc((void**)&cudaCities, sizeof(int)*specimenbits*specimenbits);
	cudaMemcpy(cudaCities, cities, sizeof(int)*specimenbits*specimenbits, cudaMemcpyHostToDevice);

	//! Настройка библиотеки cuRand
	curandState *states = 0;
	cudaMalloc((void**)&states, sizeof(curandState) * length);

	setupStates<<<BLOCKS, THREADS>>>(states, length, rand() % RAND_MAX_GA, rand() % RAND_MAX_GA);

	//! Копируем список городов в константную память
	int city[specimenbits], i;
	for(i = 0; i < specimenbits; ++i) {city[i] = i;}
	cudaMemcpyToSymbol(cudaCity, &city, specimenbits*sizeof(int), 0, cudaMemcpyHostToDevice);

	//! Создание популяций
	specimen *devPopulation = 0, *devNewPopulation = 0;
	cudaMalloc((void**)&devPopulation, sizeof(specimen) * length);
	cudaMalloc((void**)&devNewPopulation, sizeof(specimen) * length);
	cudaThreadSynchronize();

	initPopulation<<<BLOCKS, THREADS>>>(devPopulation, length, states);
	cudaThreadSynchronize();

	//! Итерации
	for(i = 0; i < 30; ++i){

		countFitness<<<BLOCKS, THREADS>>>(devPopulation, cudaCities, length);
		newGeneration<<<BLOCKS, HALF_THREADS>>>(devPopulation, devNewPopulation, length, states);
		cudaThreadSynchronize();
		swap(devPopulation, devNewPopulation);
	}

	countFitness<<<BLOCKS, THREADS>>>(devPopulation, cudaCities, length);
	cudaThreadSynchronize();
	findBestSpecimen<<<1, THREADS>>>(devPopulation, length);

	specimen best;
	cudaMemcpy(&best, &devPopulation[0], sizeof(specimen), cudaMemcpyDeviceToHost);

	printf("%d ", best.fitness);

	cudaFree(cudaCities);
	cudaFree(states);
	cudaFree(devPopulation);
	cudaFree(devNewPopulation);
}
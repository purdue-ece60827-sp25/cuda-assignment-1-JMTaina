
#include "cudaLib.cuh" 

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		y[i] = scale * x[i] + y[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	/*	Generate two random vectors X and Y of vectorSize each */
	uint64_t sizeRaw = (uint64_t) vectorSize * sizeof(float);
	if (sizeRaw > 2147483647) {
		printf("Size larger than largest signed int value, aborting\n");
		return -1;
	}
	int size = (int)sizeRaw;
	
	//	Allocate input vectors X and Y in host memory
	float* x; /* CONST */
	float* y; /* CONST */
	float* z; /* output vector use to verify z[i] = a*x[i] + y[i] */

	x = (float *) malloc(size);
	y = (float *) malloc(size);
	z = (float *) malloc(size);

	if (x == NULL || y == NULL || z == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	//	Initialize input vectors
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);
	dbprintf("Successfully allocated input vectors in CPU memory!\n");

	//	Allocate vectors in device memory
	float* x_d; /* CONST */
	float* z_d; /* MUTABLE: y[i]+ = a*x[i] */

	gpuErrchk(cudaMalloc((void **) &x_d, size));
	gpuErrchk(cudaMalloc((void **) &z_d, size));
	dbprintf("Successfully malloced space in GPU\n");

	//	Pre-invoke: Transfer X and Y to device memory
	gpuErrchk(cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(z_d, y, size, cudaMemcpyHostToDevice));
	float scale = 2.0f;
	//float scale = 2.1f;
	//float scale = 2.18735f;
	//float scale = 4294967296.0f;
	//float scale = 536870911.0f;
	//float scale = 1.0f;
	//float scale = 0.0009765625f;
	//float scale = 0.001035553f;
	//float scale = 1024.0f;
	//float scale = 14.34f;
	//float scale = 0.041f;
	dbprintf("Successfully memcpy'd from host to device!\n");

	#ifndef DEBUG_PRINT_DISABLE
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" x   = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", x[i]);
		}
		printf(" ... }\n");
		printf(" y   = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	//	Invoke kernel...
	int threadsPerBlock = 256;
	int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
	saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(x_d, z_d, scale, vectorSize);

	//	Post-invoke: Transfer new Y to host memory
	gpuErrchk(cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost));

	
	#ifndef DEBUG_PRINT_DISABLE
		printf(" z   = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", z[i]);
		}
		printf(" ... }\n");
		printf(" err < { ");
		for (int i = 0; i < 5; ++i) {
			printf("%.1e, ", std::numeric_limits<float>::epsilon() * std::fmax(std::fabs(z[i]), std::fabs(scale * x[i] + y[i])));
		}
		printf(" ... }\n");
	#endif

	//	Verify results
	int errorCount = verifyVector(x, y, z, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	//	Free device memory, host memory
	gpuErrchk(cudaFree(x_d));
	gpuErrchk(cudaFree(z_d));
	
	free(x);
	free(y);
	free(z);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;
	float x;
	float y;
	uint64_t pSum = 0;

	if (id < pSumSize) {
		//	Setup RNG
		curandState_t rng;
		curand_init(1234, id, 0, &rng);

		//	Sum up hit counts for each thread
		for (uint64_t i = 0; i < sampleSize; i++) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			if ((x*x + y*y) < 1) {
				++pSum;
			}
		}
		pSums[id] = pSum;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here

	//	Compare given thread count to device max thread count, use whichever is lower
	/* GIVEN THREAD COUNT */
	uint64_t threadCount = generateThreadCount;

	/* DEVICE MAX THREAD COUNT */
	int numBlocks;
	int threadsPerBlock = 256;

	int device;
	cudaDeviceProp prop;
	int activeThreads;
	int maxThreads;
	int maxSMs;

	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks,
		generatePoints,
		threadsPerBlock,
		0);

	activeThreads = numBlocks * threadsPerBlock;
	maxThreads = prop.maxThreadsPerMultiProcessor;
	maxSMs = prop.multiProcessorCount;

	dbprintf("%d active threads for kernel 'generatePoints'\n", activeThreads);
	dbprintf("%d total threads per multiprocessor\n", maxThreads);
	dbprintf("%d SMs in the V100 GPU\n", maxSMs);
	dbprintf("%d total concurrent threads for kernel 'generatePoints'\n", activeThreads * maxSMs);

	/* PERFORMANCE: Warn user if GPU occupancy is saturated */
	if (threadCount > (uint64_t)(activeThreads * maxSMs)) {
		dbprintf("GPU occupancy is saturated, generateThreadCount exceeds max concurrent threads for GPU\n");
	}

	//	Allocate array to track each thread's hit count, in CPU and GPU
	//	TODO: Use reduceCounts() to reduce size of transfer from thread count to 1
	uint64_t* hitCounts;
	hitCounts = (uint64_t *) malloc(threadCount * sizeof(uint64_t));

	if (hitCounts == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}
	uint64_t* hitCounts_d;
	gpuErrchk(cudaMalloc((void **) &hitCounts_d, threadCount * sizeof(uint64_t)));
	dbprintf("Successfully malloced space in GPU\n");

	//	Invoke kernel...
	// int threadsPerBlock = 256;
	int blocksPerGrid = ((int)(threadCount) + threadsPerBlock - 1) / threadsPerBlock;
	generatePoints<<<blocksPerGrid, threadsPerBlock>>>(hitCounts_d, threadCount, sampleSize);

	//	Sum up hit counts from each thread, then calculate pi
	//	TODO: Use reduceCounts() to sum up hit counts in GPU before memcpy grand total to CPU
	gpuErrchk(cudaMemcpy(hitCounts, hitCounts_d, threadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	
	uint64_t totalHitCount = 0;
	for (uint64_t idx = 0; idx < threadCount; idx++) {
		totalHitCount += hitCounts[idx];
	}
	approxPi = ((double)totalHitCount / sampleSize) / threadCount;
	approxPi = approxPi * 4.0f;

	//	Free device memory, host memory
	gpuErrchk(cudaFree(hitCounts_d));

	free(hitCounts);

	//	End of code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}

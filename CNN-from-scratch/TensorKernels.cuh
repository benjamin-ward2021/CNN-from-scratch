#pragma once

#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace TensorKernels {
	/// <summary>
	/// The CUDA kernel used to calculate matrix multiplications. 
	/// Credit to https://github.com/tgautam03/CUDA-C/blob/master/05_tiled_mat_mul/tiled_mat_mul_gpu.cu
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="deviceData">Pointer to the data for this object.</param>
	/// <param name="deviceOtherData">Pointer to the data for the other tensor object.</param>
	/// <param name="deviceProductData">Pointer to the data for the product tensor object.</param>
	/// <param name="M">The unique dimension of this matrix.</param>
	/// <param name="K">The shared dimension between this and other.</param>
	/// <param name="N">The unique dimension of the other matrix.</param>
	/// <param name="tileSize">The size .</param>
	template <typename T>
	__global__ void matrixMultiplyGPUKernel(const T *deviceData, const T *deviceOtherData, T *deviceProductData, int M, int K, int N, const int tileSize) {
		// Ensure that tileSize = BLOCK_SIZE
		assert(tileSize == blockDim.x && tileSize == blockDim.y);

		// Details regarding this thread
		int by = blockIdx.y;
		int bx = blockIdx.x;

		int ty = threadIdx.y;
		int tx = threadIdx.x;

		// Working on C[i,j]
		int i = tileSize * by + ty;
		int j = tileSize * bx + tx;

		// Declare dynamic shared memory
		extern __shared__ __align__(sizeof(T)) T sharedMemory[];

		// Partition the shared memory
		T *sharedData = sharedMemory;
		T *sharedOtherData = &sharedMemory[tileSize * tileSize];

		T value = static_cast<T>(0);
		for (int phase = 0; phase < (K + tileSize - 1) / tileSize; phase++)
		{
			// Load Tiles into shared memory
			if (i < M && phase * tileSize + tx < K) {
				sharedData[ty * tileSize + tx] = deviceData[i * K + phase * tileSize + tx];
			}
			else {
				sharedData[ty * tileSize + tx] = static_cast<T>(0);
			}

			if (phase * tileSize + ty < K && j < N) {
				sharedOtherData[ty * tileSize + tx] = deviceOtherData[(phase * tileSize + ty) * N + j];
			}
			else {
				sharedOtherData[ty * tileSize + tx] = static_cast<T>(0);
			}
			__syncthreads();

			// Dot product
			for (int k = 0; k < tileSize; k++) {
				value += sharedData[ty * tileSize + k] * sharedOtherData[k * tileSize + tx];
			}
			__syncthreads();
		}

		// Assigning calculated value
		if (i < M && j < N) {
			deviceProductData[i * N + j] = value;
		}
	}
}
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

// TODO: These are the naive implementations of these operations. Very inefficient. Doesn't use shared memory.
// Very intuitive, though
namespace Conv2DKernels {
	/// <summary>
	/// Performs forward propagation for a conv2d layer.
	/// </summary>
	template <typename T>
	__global__ void forwardGPUKernel(const T *inputs, const T *weights, const T *biases, T *outputs,
		int batchSize, int inputChannels, int inputHeight, int inputWidth,
		int outputChannels, int kernelSize, int stride, int padding,
		int outputHeight, int outputWidth) {

		const int outputsIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (outputsIndex >= batchSize * outputChannels * outputHeight * outputWidth) {
			return;
		}

		// Unravel 4D indices [N, C, H, W]
		const int batchIndex = outputsIndex / (outputChannels * outputHeight * outputWidth);
		const int outputChannelIndex = (outputsIndex / (outputHeight * outputWidth)) % outputChannels;
		const int outputHeightIndex = (outputsIndex / outputWidth) % outputHeight;
		const int outputWidthIndex = outputsIndex % outputWidth;

		T sum = biases ? biases[outputChannelIndex] : static_cast<T>(0);

		for (int inputChannelIndex = 0; inputChannelIndex < inputChannels; inputChannelIndex++) {
			for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize; kernelHeightIndex++) {
				for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize; kernelWidthIndex++) {
					const int inputHeightIndex = outputHeightIndex * stride + kernelHeightIndex - padding;
					const int inputWidthIndex = outputWidthIndex * stride + kernelWidthIndex - padding;

					if (inputHeightIndex >= 0 && inputHeightIndex < inputHeight && inputWidthIndex >= 0 && inputWidthIndex < inputWidth) {
						const int inputsIndex = batchIndex * inputChannels * inputHeight * inputWidth +
							inputChannelIndex * inputHeight * inputWidth +
							inputHeightIndex * inputWidth +
							inputWidthIndex;

						const int weightsIndex = outputChannelIndex * inputChannels * kernelSize * kernelSize +
							inputChannelIndex * kernelSize * kernelSize +
							kernelHeightIndex * kernelSize +
							kernelWidthIndex;

						sum += inputs[inputsIndex] * weights[weightsIndex];
					}
				}
			}
		}

		outputs[outputsIndex] = sum;
	}

	/// <summary>
	/// Finds the gradient with respect to the weights for a conv2d layer.
	/// </summary>
	template <typename T>
	__global__ void gradWrtWeightsGPUKernel(const T *inputs, const T *gradWrtOutputs, T *gradWrtWeights,
		int batchSize, int inputChannels, int inputHeight, int inputWidth,
		int outputChannels, int kernelSize, int stride, int padding,
		int outputHeight, int outputWidth) {

		// Each thread handles one weight gradient element [outputChannelIndex, inputChannelIndex, kernelHeightIndex, kernelWidthIndex]
		const int outputChannelIndex = blockIdx.x;
		const int inputChannelIndex = blockIdx.y;
		const int kernelHeightIndex = blockIdx.z / kernelSize;
		const int kernelWidthIndex = blockIdx.z % kernelSize;

		if (outputChannelIndex >= outputChannels || inputChannelIndex >= inputChannels || kernelHeightIndex >= kernelSize || kernelWidthIndex >= kernelSize) {
			return;
		}

		T sum = static_cast<T>(0);

		for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
			for (int outputHeightIndex = 0; outputHeightIndex < outputHeight; outputHeightIndex++) {
				for (int outputWidthIndex = 0; outputWidthIndex < outputWidth; outputWidthIndex++) {
					const int inputHeightIndex = outputHeightIndex * stride + kernelHeightIndex - padding;
					const int inputWidthIndex = outputWidthIndex * stride + kernelWidthIndex - padding;

					if (inputHeightIndex >= 0 && inputHeightIndex < inputHeight && inputWidthIndex >= 0 && inputWidthIndex < inputWidth) {
						const int inputsIndex = batchIndex * inputChannels * inputHeight * inputWidth +
							inputChannelIndex * inputHeight * inputWidth +
							inputHeightIndex * inputWidth +
							inputWidthIndex;

						const int gradWrtOutputsIndex = batchIndex * outputChannels * outputHeight * outputWidth +
							outputChannelIndex * outputHeight * outputWidth +
							outputHeightIndex * outputWidth +
							outputWidthIndex;

						sum += inputs[inputsIndex] * gradWrtOutputs[gradWrtOutputsIndex];
					}
				}
			}
		}

		const int gradWrtWeightsIndex = outputChannelIndex * inputChannels * kernelSize * kernelSize +
			inputChannelIndex * kernelSize * kernelSize +
			kernelHeightIndex * kernelSize +
			kernelWidthIndex;

		gradWrtWeights[gradWrtWeightsIndex] = sum;
	}


	/// <summary>
	/// Finds the gradient with respect to the inputs for a conv2d layer.
	/// </summary>
	template <typename T>
	__global__ void gradWrtInputsGPUKernel(const T *gradWrtOutputs, const T *weights, T *gradWrtInputs,
		int batchSize, int inputChannels, int inputHeight, int inputWidth,
		int outputChannels, int kernelSize, int stride, int padding,
		int outputHeight, int outputWidth) {

		// Each thread handles one input gradient element [batchIndex, inputChannelIndex, inputHeightIndex, inputWidthIndex]
		const int gradWrtInputsIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (gradWrtInputsIndex >= batchSize * inputChannels * inputHeight * inputWidth) {
			return;
		}

		// Unravel 4D indices [N, C, H, W]
		const int batchIndex = gradWrtInputsIndex / (inputChannels * inputHeight * inputWidth);
		const int inputChannelIndex = (gradWrtInputsIndex / (inputHeight * inputWidth)) % inputChannels;
		const int inputHeightIndex = (gradWrtInputsIndex / inputWidth) % inputHeight;
		const int inputWidthIndex = gradWrtInputsIndex % inputWidth;

		T sum = static_cast<T>(0);

		for (int outputChannelIndex = 0; outputChannelIndex < outputChannels; outputChannelIndex++) {
			for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize; kernelHeightIndex++) {
				for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize; kernelWidthIndex++) {
					const int outputHeightIndex = (inputHeightIndex - kernelHeightIndex + padding) / stride;
					const int outputWidthIndex = (inputWidthIndex - kernelWidthIndex + padding) / stride;

					if (outputHeightIndex >= 0 && outputHeightIndex < outputHeight &&
						outputWidthIndex >= 0 && outputWidthIndex < outputWidth &&
						(inputHeightIndex - kernelHeightIndex + padding) % stride == 0 &&
						(inputWidthIndex - kernelWidthIndex + padding) % stride == 0) {

						const int weightsIndex = outputChannelIndex * inputChannels * kernelSize * kernelSize +
							inputChannelIndex * kernelSize * kernelSize +
							kernelHeightIndex * kernelSize +
							kernelWidthIndex;

						const int gradWrtOutputsIndex = batchIndex * outputChannels * outputHeight * outputWidth +
							outputChannelIndex * outputHeight * outputWidth +
							outputHeightIndex * outputWidth +
							outputWidthIndex;

						sum += weights[weightsIndex] * gradWrtOutputs[gradWrtOutputsIndex];
					}
				}
			}
		}

		gradWrtInputs[gradWrtInputsIndex] = sum;
	}
}
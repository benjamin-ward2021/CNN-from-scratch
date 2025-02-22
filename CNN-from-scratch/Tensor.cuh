#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <string>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "TensorKernels.cuh"

/// <summary>
/// A multidimensional array.
/// </summary>
/// <typeparam name="T"></typeparam>
template<typename T>
class Tensor {
public:
	// All types of tensors are friends with all other types of tensors, and can access their private variables.
	// (Ex. Tensor<float> can access dims of a Tensor<int>).
	template <typename U>
	friend class Tensor;

	/// <summary>
	/// Default constructor does nothing.
	/// </summary>
	Tensor() {}

	/// <summary>
	/// Initializes data with each element being the default value.
	/// (Ex. 0 for int).
	/// </summary>
	/// <param name="dims"></param>
	Tensor(const std::vector<int> &dims) : dims(dims) {
		assert(dims.size() > 0);
		coordinateConversionLookupTable = createCoordinateConversionLookupTable(dims);
		assert(coordinateConversionLookupTable.size() > 0);
		int size = getSizeFromDims(dims);
		assert(size > 0);
		data = std::vector<T>(size);
	}

	/// <summary>
	/// Initializes data using the given vector.
	/// </summary>
	/// <param name="dims"></param>
	Tensor(const std::vector<T> &data, const std::vector<int> &dims) : data(data), dims(dims) {
		assert(dims.size() > 0);
		coordinateConversionLookupTable = createCoordinateConversionLookupTable(dims);
		int size = getSizeFromDims(dims);

		// Verify that the data passed in matches the expected size
		assert(data.size() == size);
	}

	bool operator==(const Tensor<T> &other) const {
		return data == other.data && dims == other.dims && coordinateConversionLookupTable == other.coordinateConversionLookupTable;
	}

	bool operator!=(const Tensor<T> &other) const {
		return !(*this == other);
	}

	/// <summary>
	/// Creates a new tensor that conains a statically casted version of the data.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <returns></returns>
	template <typename U>
	[[nodiscard]]
	Tensor<U> convert() const {
		Tensor<U> converted(dims);
		for (int i = 0; i < data.size(); i++) {
			converted.data[i] = static_cast<U>(data[i]);
		}

		return converted;
	}

	// TODO: Should there also be a constructor for moving data instead of copying it?
	// See https://en.cppreference.com/w/cpp/utility/move

	/// <summary>
	/// Gets an element from data. Excluded indices are assumed to be zero.
	/// (Ex. if indices = { 2,3 }, get the element at row 2 column 3. Equivalent to { 2,3,0 } for a 3d tensor).
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The element at the specified indices.</returns>
	[[nodiscard]]
	T get(const std::vector<int> &indices) const {
		int flattenedIndex = getFlattenedIndex(indices);
		return data[flattenedIndex];
	}

	/// <summary>
	/// Sets the element to value. Excluded indices are assumed to be zero.
	/// </summary>
	/// <param name="indices"></param>
	/// <param name="value"></param>
	void set(const std::vector<int> &indices, T value) {
		int flattenedIndex = getFlattenedIndex(indices);
		data[flattenedIndex] = value;
	}

	/// <summary>
	/// Sets all elements in the tensor to a random value using the normal distribution and random engine.
	/// Parameters are passed by mutable reference and can be changed.
	/// </summary>
	/// <param name="normalDistribution"></param>
	/// <param name="randomEngine"></param>
	void setToRandom(std::normal_distribution<T> normalDistribution, std::default_random_engine randomEngine) {
		for (int i = 0; i < data.size(); i++) {
			data[i] = normalDistribution(randomEngine);
		}
	}

	/// <summary>
	/// Sets all elements in a tensor to a random value using the uniform distribution and random engine.
	/// Parameters are passed by mutable reference and can be changed.
	/// </summary>
	/// <param name="uniformDistribution"></param>
	/// <param name="randomEngine"></param>
	void setToRandom(std::uniform_real_distribution<T> &uniformDistribution, std::default_random_engine &randomEngine) {
		for (int i = 0; i < data.size(); i++) {
			data[i] = uniformDistribution(randomEngine);
		}
	}

	/// <summary>
	/// Sets all elements to 0.
	/// </summary>
	void setToZero() {
		for (int i = 0; i < data.size(); i++) {
			data[i] = static_cast<T>(0);
		}
	}

	/// <summary>
	/// Gets the dimensions
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	std::vector<int> getDims() const {
		return dims;
	}

	/// <summary>
	/// Reshapes the tensor.
	/// </summary>
	/// <param name="dims"></param>
	void reinterpretDims(const std::vector<int> &dims) {
		coordinateConversionLookupTable = createCoordinateConversionLookupTable(dims);
		int size = getSizeFromDims(dims);
		assert(size == data.size());
		this->dims = dims;
	}

	/// <summary>
	/// Flattens one dimension of the tensor.
	/// (Ex. { 2,3,4,5 } becomes { 2*3,4,5 } = { 6,4,5 }).
	/// </summary>
	void flattenOnce() {
		assert(dims.size() > 1);
		std::vector<int> newDims = dims;
		newDims[0] *= newDims[1];
		newDims.erase(newDims.begin() + 1);
		reinterpretDims(newDims);
	}

	/// <summary>
	/// Flattens the tensor to 1d.
	/// (Ex. { 2,3,4,5 } becomes { 2*3*4*5 } = { 120 }).
	/// </summary>
	void flattenFully() {
		reinterpretDims({ static_cast<int>(data.size()) });
	}

	/// <summary>
	/// Flattens the tensor, ignoring the first dimension.
	/// This is useful since the first dimension is often the batch size.
	/// (Ex. { 2,3,4,5 } becomes { 2,3*4*5 } = { 2,60 }).
	/// </summary>
	void reverseFlattenTo2d() {
		assert(dims.size() > 1);
		reinterpretDims({ dims[0],static_cast<int>(data.size()) / dims[0] });
	}

	/// <summary>
	/// Broadcasts a tensor by repeating dimensions.
	/// For visualizations I recommend https://medium.com/@hunter-j-phillips/a-simple-introduction-to-broadcasting-db8e581368b3
	/// </summary>
	/// <param name="broadcastedDims"></param>
	/// <returns>The expanded tensor.</returns>
	[[nodiscard]]
	Tensor<T> broadcast(const std::vector<int> &broadcastedDims) const {
		assert(dims.size() > 0 && broadcastedDims.size() > 0);
		assert(broadcastedDims.size() >= dims.size());

		std::vector<int> paddedDims = dims;
		// Add dummy dimensions
		while (paddedDims.size() < broadcastedDims.size()) {
			paddedDims.insert(paddedDims.begin(), 1);
		}

		assert(paddedDims.size() == broadcastedDims.size());

		for (int i = 0; i < broadcastedDims.size(); i++) {
			assert(paddedDims[i] == 1 || paddedDims[i] == broadcastedDims[i]);
		}

		std::vector<int> paddedLookupTable = createCoordinateConversionLookupTable(paddedDims);
		std::vector<int> broadcastedLookupTable = createCoordinateConversionLookupTable(broadcastedDims);

		Tensor<T> broadcasted(broadcastedDims);

		for (int i = 0; i < broadcasted.data.size(); i++) {
			int dataIndex = 0;
			for (int dim = 0; dim < broadcastedDims.size(); dim++) {
				int coordinate = paddedDims[dim] == 1 ? 0 : (i / broadcastedLookupTable[dim]) % paddedDims[dim];
				dataIndex += coordinate * paddedLookupTable[dim];
			}

			assert(dataIndex < data.size());
			broadcasted.data[i] = data[dataIndex];
		}

		return broadcasted;
	}

	/// <summary>
	/// Provides a way to get batchSize random indices of samples. 
	/// Used in combination with getBatch, so inputs and targets can share the same mapping.
	/// </summary>
	/// <param name="batchSize"></param>
	/// <param name="randomEngine"></param>
	/// <returns></returns>
	[[nodiscard]]
	std::vector<int> getRandomIndices(int batchSize, std::default_random_engine &randomEngine) const {
		const int totalNumSamples = dims[0];
		assert(batchSize <= totalNumSamples);
		std::vector<int> indices(totalNumSamples);
		for (int i = 0; i < totalNumSamples; i++) {
			indices[i] = i;
		}

		std::shuffle(indices.begin(), indices.end(), randomEngine);

		return std::vector<int>(indices.begin(), indices.begin() + batchSize);
	}

	/// <summary>
	/// Gets a batch of samples based on their indices.
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>A tensor with dims = { batchSize,dims[1],dims[2],dims[3]... }</returns>
	[[nodiscard]]
	Tensor<T> getBatch(const std::vector<int> &indices) const {
		const int batchSize = static_cast<int>(indices.size());
		const int totalNumSamples = dims[0];
		assert(batchSize <= totalNumSamples);
		std::vector<int> batchDims = dims;
		batchDims[0] = static_cast<int>(indices.size());
		Tensor<T> batch(batchDims);
		assert(coordinateConversionLookupTable == batch.coordinateConversionLookupTable);
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < coordinateConversionLookupTable[0]; j++) {
				batch.data[i * coordinateConversionLookupTable[0] + j] = data[indices[i] * coordinateConversionLookupTable[0] + j];
			}
		}

		return batch;
	}

	/// <summary>
	/// Gets a single sample from the tensor, where the first dimension is the batch.
	/// </summary>
	/// <param name="index">The index of the sample relative to the batch.</param>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> getSampleAt(int index, bool keepDummyDim = false) const {
		assert(dims.size() > 0 && index < dims[0]);
		std::vector<int> sampleDims = dims;
		if (keepDummyDim) {
			sampleDims[0] = 1;
		}

		else {
			sampleDims.erase(sampleDims.begin());
		}

		Tensor<T> sample(sampleDims);
		const int sampleSize = coordinateConversionLookupTable[0];
		for (int i = 0; i < sampleSize; i++) {
			sample.data[i] = data[sampleSize * index + i];
		}

		return sample;
	}

	/// <summary>
	/// Converts a one-hot encoded label to an int.
	/// </summary>
	/// <param name="index">The index of the sample relative to the batch.</param>
	/// <returns></returns>
	[[nodiscard]]
	int convertLabelToInt(int index) const {
		assert(dims.size() > 0 && index < dims[0]);

		const int sampleSize = coordinateConversionLookupTable[0];
		for (int i = 0; i < sampleSize; i++) {
			if (data[sampleSize * index + i] == 1) {
				return i;
			}
		}

		throw std::runtime_error("One-hot label has no one");
	}

	/// <summary>
	/// Multiplies two matrices together. The algorithm uses tiling, which offers significant performance gains 
	/// due to more efficient cache use. See https://marek.ai/matrix-multiplication-on-cpu.html for more information.
	/// </summary>
	/// <param name="other"></param>
	/// <param name="tileSize">112 chosen as default based on performance. If working with floats instead of doubles, you could try 160.</param>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> matrixMultiply(const Tensor<T> &other, const int tileSize = 112) const {
		assert(dims.size() == 2 && other.dims.size() == 2);
		assert(dims[1] == other.dims[0]);
		// this = M * K, other = K * N, product = M * N
		int M = dims[0];
		int N = other.dims[1];
		int K = dims[1];

		Tensor<T> product({ M, N });

		for (int i = 0; i < M; i += tileSize) {
			int i_max = std::min(i + tileSize, M);
			for (int j = 0; j < N; j += tileSize) {
				int j_max = std::min(j + tileSize, N);
				for (int k = 0; k < K; k += tileSize) {
					int k_max = std::min(k + tileSize, K);
					for (int ii = i; ii < i_max; ii++) {
						for (int kk = k; kk < k_max; kk++) {
							T a_val = data[ii * K + kk];
							for (int jj = j; jj < j_max; jj++) {
								product.data[ii * N + jj] += a_val * other.data[kk * N + jj];
							}
						}
					}
				}
			}
		}

		return product;
	}

	/// <summary>
	/// Performs matrix multiplication on the GPU using CUDA. 
	/// Big credit to https://0mean1sigma.com/chapter-4-memory-coalescing-and-tiled-matrix-multiplication/,
	/// I recommend his whole video series found here https://www.youtube.com/@0mean1sigma/videos
	/// </summary>
	/// <param name="other"></param>
	/// <param name="tileSize">32 is the max, since CUDA supports 1024 threads per block (32 * 32 = 1024).</param>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> matrixMultiplyGPU(const Tensor<T> &other, const int tileSize = 32) const {
		assert(dims.size() == 2 && other.dims.size() == 2);
		assert(dims[1] == other.dims[0]);
		// The max threads per block that CUDA allows is 1024
		assert(tileSize * tileSize <= 1024);
		// this = M * K, other = K * N, product = M * N
		int M = dims[0];
		int K = dims[1];
		int N = other.dims[1];

		Tensor<T> product({ M, N });

		// Allocate and copy this objects data to GPU memory
		T *deviceData;
		cudaError_t deviceDataMallocStatus = cudaMalloc((void **)&deviceData, M * K * sizeof(T));
		checkCuda(deviceDataMallocStatus);
		cudaError_t deviceDataMemcpyStatus = cudaMemcpy(deviceData, data.data(), M * K * sizeof(T), cudaMemcpyHostToDevice);
		checkCuda(deviceDataMemcpyStatus);

		// Allocate and copy the other tensors data to GPU memory
		T *deviceOtherData;
		cudaError_t deviceOtherDataMallocStatus = cudaMalloc((void **)&deviceOtherData, K * N * sizeof(T));
		checkCuda(deviceOtherDataMallocStatus);
		cudaError_t deviceOtherDataMemcpyStatus = cudaMemcpy(deviceOtherData, other.data.data(), K * N * sizeof(T), cudaMemcpyHostToDevice);
		checkCuda(deviceOtherDataMemcpyStatus);

		// Allocate space for the result of the matrix multiplication
		T *deviceProductData;
		cudaError_t deviceProductDataMallocStatus = cudaMalloc((void **)&deviceProductData, M * N * sizeof(T));
		checkCuda(deviceProductDataMallocStatus);

		// Define arguments for kernel
		dim3 dim_block(tileSize, tileSize, 1);
		dim3 dim_grid((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize, 1);
		size_t sharedMemorySize = 2 * tileSize * tileSize * sizeof(T);

		// Call kernel
		TensorKernels::matrixMultiplyGPUKernel<<<dim_grid, dim_block, sharedMemorySize>>>(deviceData, deviceOtherData, deviceProductData, M, K, N, tileSize);
		// Check for kernel launch errors
		cudaError_t kernelLaunchStatus = cudaGetLastError();
		checkCuda(kernelLaunchStatus);

		// Copy back results into product
		cudaError_t deviceProductDataMemcpyStatus = cudaMemcpy(product.data.data(), deviceProductData, M * N * sizeof(T), cudaMemcpyDeviceToHost);
		checkCuda(deviceProductDataMemcpyStatus);

		// Free memory
		cudaFree(deviceData);
		cudaFree(deviceOtherData);
		cudaFree(deviceProductData);

		return product;
	}

	/// <summary>
	/// Transposes the matrix by changing the data, rather than changing the indexing.
	/// Uses a tiling approach, which improves caching hit rate.
	/// </summary>
	/// <param name="tileSize">112 chosen as default based on performance. If working with floats instead of doubles, you could try 160.</param>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> transpose(const int tileSize = 112) const {
		assert(dims.size() == 2);
		int rows = dims[0];
		int cols = dims[1];
		Tensor<T> transposed({ cols,rows });
		for (int i = 0; i < rows; i += tileSize) {
			int i_max = std::min(i + tileSize, rows);
			for (int j = 0; j < cols; j += tileSize) {
				int j_max = std::min(j + tileSize, cols);
				for (int ii = i; ii < i_max; ii++) {
					for (int jj = j; jj < j_max; jj++) {
						transposed.data[jj * rows + ii] = data[ii * cols + jj];
					}
				}
			}
		}

		return transposed;
	}

	// TODO: Going to eventually need a transposeGPU

	/// <summary>
	/// Sums the rows of a matrix together.
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> matrixRowSum() const {
		assert(dims.size() == 2);
		int rows = dims[0];
		int cols = dims[1];
		Tensor<T> sum({ rows });
		for (int i = 0; i < rows; i++) {
			T value = 0;
			for (int j = 0; j < cols; j++) {
				value += data[i * cols + j];
			}

			sum.data[i] = value;
		}

		return sum;
	}
	
	/// <summary>
	/// Sums the columns of a matrix together.
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> matrixColumnSum() const {
		assert(dims.size() == 2);
		return this->transpose().matrixRowSum();
	}

	// TODO: Both sum and max share most of their code. Maybe there can be a private method "reduce" that takes a function pointer
	// for whatever operation needs to be performed.

	/// <summary>
	/// Sums over an axis in the tensor. 
	/// </summary>
	/// <param name="dim">The index of the dimension, starting at 0.</param>
	/// <param name="keepDummyDim">Whether to keep the dimension as 1, or remove it entirely.</param>
	/// <returns>New tensor with values summed along the dimension</returns>
	[[nodiscard]]
	Tensor<T> sum(int dim, bool keepDummyDim = false) const {
		assert(dim < dims.size());

		std::vector<int> sumDims = dims;
		if (keepDummyDim) {
			sumDims[dim] = 1;
		}

		else {
			sumDims.erase(sumDims.begin() + dim);
		}

		Tensor<T> result(sumDims);

		assert(dims[dim] > 0 && coordinateConversionLookupTable[dim] > 0);
		// Number of blocks above target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then higherDimsSize = 3 * 7 = 21
		int higherDimsSize = static_cast<int>(data.size() / (dims[dim] * coordinateConversionLookupTable[dim]));
		assert(higherDimsSize > 0);

		// Size of target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then currentDimsSize = 5
		int currentDimsSize = dims[dim];
		assert(currentDimsSize > 0);

		// Number of elements in dimensions after the target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then lowerDimsSize = 2 * 4 = 8
		int lowerDimsSize = coordinateConversionLookupTable[dim];
		assert(lowerDimsSize > 0);

		for (int higherIndex = 0; higherIndex < higherDimsSize; higherIndex++) {
			for (int lowerIndex = 0; lowerIndex < lowerDimsSize; lowerIndex++) {
				int baseIndex = higherIndex * (currentDimsSize * lowerDimsSize) + lowerIndex;
				T accumulator = static_cast<T>(0);
				for (int currentIndex = 0; currentIndex < currentDimsSize; currentIndex++) {
					int dataIndex = baseIndex + currentIndex * lowerDimsSize;

					accumulator += data[dataIndex];
				}

				int resultDataIndex = higherIndex * lowerDimsSize + lowerIndex;
				result.data[resultDataIndex] = accumulator;
			}
		}

		return result;
	}

	/// <summary>
	/// Overloads the sum method. Sums the entire tensor.
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	T sum() {
		T accumulator = static_cast<T>(0);
		for (int i = 0; i < data.size(); i++) {
			accumulator += data[i];
		}

		return accumulator;
	}

	/// <summary>
	/// Maxes over an axis in the tensor. 
	/// </summary>
	/// <param name="dim">The index of the dimension, starting at 0.</param>
	/// <param name="keepDummyDim">Whether to keep the dimension as 1, or remove it entirely.</param>
	/// <returns>New tensor with max values along the dimension</returns>
	[[nodiscard]]
	Tensor<T> max(int dim, bool keepDummyDim = false) const {
		assert(dim < dims.size());

		std::vector<int> maxDims = dims;
		if (keepDummyDim) {
			maxDims[dim] = 1;
		}

		else {
			maxDims.erase(maxDims.begin() + dim);
		}

		Tensor<T> result(maxDims);

		assert(dims[dim] > 0 && coordinateConversionLookupTable[dim] > 0);
		// Number of blocks above target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then higherDimsSize = 3 * 7 = 21
		int higherDimsSize = static_cast<int>(data.size() / (dims[dim] * coordinateConversionLookupTable[dim]));
		assert(higherDimsSize > 0);

		// Size of target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then currentDimsSize = 5
		int currentDimsSize = dims[dim];
		assert(currentDimsSize > 0);

		// Number of elements in dimensions after the target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then lowerDimsSize = 2 * 4 = 8
		int lowerDimsSize = coordinateConversionLookupTable[dim];
		assert(lowerDimsSize > 0);

		for (int higherIndex = 0; higherIndex < higherDimsSize; higherIndex++) {
			for (int lowerIndex = 0; lowerIndex < lowerDimsSize; lowerIndex++) {
				int baseIndex = higherIndex * (currentDimsSize * lowerDimsSize) + lowerIndex;
				T currentMax = data[baseIndex];
				for (int currentIndex = 0; currentIndex < currentDimsSize; currentIndex++) {
					int dataIndex = baseIndex + currentIndex * lowerDimsSize;

					currentMax = std::max(currentMax, data[dataIndex]);
				}

				int resultDataIndex = higherIndex * lowerDimsSize + lowerIndex;
				result.data[resultDataIndex] = currentMax;
			}
		}

		return result;
	}

	/// <summary>
	/// Calculates the exp for each element.
	/// </summary>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> exp() const {
		Tensor<T> result(dims);
		for (int i = 0; i < data.size(); i++) {
			result.data[i] = std::exp(data[i]);
		}

		return result;
	}

	/// <summary>
	/// Calculates the log for each element.
	/// </summary>
	/// <param name="epsilon">A very small number that is added to avoid taking the log of 0.</param>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> log(double epsilon = 1e-8) const {
		Tensor<T> result(dims);
		for (int i = 0; i < data.size(); i++) {
			T dataPlusEpsilon = data[i] + static_cast<T>(epsilon);
			assert(dataPlusEpsilon > 0);
			result.data[i] = std::log(dataPlusEpsilon);
		}

		return result;
	}

	/// <summary>
	/// Adds two tensors of the same shape together.
	/// </summary>
	/// <param name="other"></param>
	/// <returns>The tensor of sums.</returns>
	[[nodiscard]]
	Tensor<T> elementwiseAdd(const Tensor<T> &other) const {
		assert(dims == other.dims);
		std::vector<T> sum(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			sum[i] = static_cast<T>(data[i] + other.data[i]);
		}
	
		return Tensor<T>(sum, dims);
	}

	/// <summary>
	/// Adds two tensors of the same shape, in place.
	/// (I.e. data += other.data).
	/// </summary>
	/// <param name="other"></param>
	void elementwiseAddInPlace(const Tensor<T> &other) {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			data[i] += other.data[i];
		}
	}

	/// <summary>
	/// Subtracts two tensors of the same shape.
	/// </summary>
	/// <param name="other"></param>
	/// <returns>The tensor of differences.</returns>
	[[nodiscard]]
	Tensor<T> elementwiseSubtract(const Tensor<T> &other) const {
		assert(dims == other.dims);
		std::vector<T> difference(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			difference[i] = static_cast<T>(data[i] - other.data[i]);
		}

		return Tensor<T>(difference, dims);
	}

	/// <summary>
	/// Subtracts two tensors of the same shape, in place.
	/// (I.e. data -= other.data).
	/// </summary>
	/// <param name="other"></param>
	void elementwiseSubtractInPlace(const Tensor<T> &other) {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			data[i] -= other.data[i];
		}
	}

	/// <summary>
	/// Multiplies two tensors of the same shape, in an elementwise way (NOT matrix multiplication).
	/// </summary>
	/// <param name="other"></param>
	/// <returns>The tensor of products.</returns>
	[[nodiscard]]
	Tensor<T> elementwiseMultiply(const Tensor<T> &other) const {
		assert(dims == other.dims);
		std::vector<T> product(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			product[i] = static_cast<T>(data[i] * other.data[i]);
		}

		return Tensor<T>(product, dims);
	}

	/// <summary>
	/// Multiplies a tensor with a scalar.
	/// </summary>
	/// <param name="multiplier"></param>
	/// <returns>The tensor of products.</returns>
	[[nodiscard]]
	Tensor<T> scalarMultiply(T multiplier) {
		Tensor<T> products(dims);
		for (int i = 0; i < data.size(); i++) {
			products.data[i] = data[i] * multiplier;
		}

		return products;
	}

	/// <summary>
	/// Divides two tensors of the same shape, in an elementwise way.
	/// </summary>
	/// <param name="other"></param>
	/// <returns>The tensor of quotients.</returns>
	[[nodiscard]]
	Tensor<T> elementwiseDivide(const Tensor<T> &other) const {
		assert(dims == other.dims);
		std::vector<T> quotient(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			quotient[i] = static_cast<T>(data[i] / other.data[i]);
		}

		return Tensor<T>(quotient, dims);
	}

	/// <summary>
	/// Divides a tensor with a scalar.
	/// </summary>
	/// <param name="divisor"></param>
	/// <returns>The tensor of quotients.</returns>
	[[nodiscard]]
	Tensor<T> scalarDivide(T divisor) {
		Tensor<T> quotients(dims);
		for (int i = 0; i < data.size(); i++) {
			quotients.data[i] = data[i] / divisor;
		}

		return quotients;
	}
	
	/// <summary>
	/// </summary>
	/// <returns>A tensor where data is clamped to 0 if it is negative.</returns>
	[[nodiscard]]
	Tensor<T> relu() const {
		Tensor<T> ret(dims);
		for (int i = 0; i < data.size(); i++) {
			ret.data[i] = std::max(static_cast<T>(0), data[i]);
		}

		return ret;
	}

	/// <summary>
	/// </summary>
	/// <returns>A tensor whose ith entry is 1 if data[i] > 0, and 0 otherwise.</returns>
	[[nodiscard]]
	Tensor<T> reluDerivative() const {
		Tensor<T> ret(dims);
		for (int i = 0; i < data.size(); i++) {
			ret.data[i] = data[i] > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
		}

		return ret;
	}

	/// <summary>
	/// Performs a softmax calculation over a dimension, where the outputs are the corresponding probability distribution.
	/// </summary>
	/// <param name="dim"></param>
	/// <returns></returns>
	[[nodiscard]]
	Tensor<T> softmax(int dim) const {
		// Subtract the maxes to have better stability
		Tensor<T> maxes = max(dim, true);
		Tensor<T> maxSubtracted = elementwiseSubtract(maxes.broadcast(dims));
		Tensor<T> exponentiated = maxSubtracted.exp();
		Tensor<T> sums = exponentiated.sum(dim, true);
		Tensor<T> result = exponentiated.elementwiseDivide(sums.broadcast(dims));

		return result;
	}

	/// <summary>
	/// Prints a 2d tensor to console. Dimensions are represented as { row,col }.
	/// </summary>
	void print2d(std::string delimiter = " ") const {
		assert(dims.size() == 2);
		for (int row = 0; row < dims[0]; row++) {
			for (int col = 0; col < dims[1]; col++) {
				std::cout << get({ row,col }) << delimiter;
			}
			std::cout << std::endl;
		}
	}

	/// <summary>
	/// Prints a 2d tensor to console. Meant for visualizing images from the MNIST dataset.
	/// </summary>
	/// <param name="threshold"></param>
	void print2dThreshold(T threshold) const {
		assert(dims.size() == 2);
		for (int row = 0; row < dims[0]; row++) {
			for (int col = 0; col < dims[1]; col++) {
				T value = get({ row,col });
				if (value >= threshold) {
					std::cout << "#";
				}

				else {
					std::cout << " ";
				}
			}
			std::cout << std::endl;
		}
	}

	/// <summary>
	/// Prints a 3d tensor to console. Dimensions are represented as { row,col,depth }.
	/// </summary>
	void print3d(std::string delimiter = " ") const {
		assert(dims.size() == 3);
		for (int depth = 0; depth < dims[2]; depth++) {
			std::cout << "Depth: " << depth << std::endl;
			for (int row = 0; row < dims[0]; row++) {
				for (int col = 0; col < dims[1]; col++) {
					std::cout << get({ row,col,depth}) << delimiter;
				}
				std::cout << std::endl;
			}
		}
	}

private:
	/// <summary>
	/// A vector containing the flattened data from a multidimensional tensor.
	/// </summary>
	std::vector<T> data;
	/// <summary>
	/// The dimensions of the tensor. 
	/// (Ex. { 2,3 } for a matrix with 2 rows and 3 columns).
	/// </summary>
	std::vector<int> dims;
	/// <summary>
	/// Lookup table used for converting multidimensional coordinates to a flattened representation.
	/// </summary>
	std::vector<int> coordinateConversionLookupTable;

	/// <summary>
	/// Creates the coordinate conversion table, and returns the total size of data.
	/// </summary>
	/// <param name="dims"></param>
	/// <returns>The total size of the data vector.</returns>
	[[nodiscard]]
	std::vector<int> createCoordinateConversionLookupTable(const std::vector<int> &dims) const {
		assert(dims.size() > 0);

		std::vector<int> lookupTable(dims.size());

		// Total number of elements in data
		int size = 1;
		for (int i = 0; i < dims.size(); i++) {
			int currentIndex = static_cast<int>(dims.size()) - 1 - i;
			lookupTable[currentIndex] = size;
			size *= dims[currentIndex];
		}

		return lookupTable;
	}

	/// <summary>
	/// Multiplies the dims together to get the total size.
	/// </summary>
	/// <param name="dims"></param>
	/// <returns></returns>
	[[nodiscard]]
	int getSizeFromDims(const std::vector<int> &dims) {
		assert(dims.size() > 0);

		int size = 1;
		for (int i = 0; i < dims.size(); i++) {
			size *= dims[i];
		}

		return size;
	}

	/// <summary>
	/// Converts a multidimensional set of indices to the equivalent flattened index
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The flattened index</returns>
	[[nodiscard]]
	int getFlattenedIndex(const std::vector<int> &indices) const {
		int flattenedIndex = 0;
		for (int i = 0; i < indices.size(); i++) {
			// Make sure the index isn't out of bounds
			assert(indices[i] < dims[i]);
			flattenedIndex += coordinateConversionLookupTable[i] * indices[i];
		}

		return flattenedIndex;
	}

	/// <summary>
	/// Checks the CUDA error code, and exits if it wasn't successful.
	/// Credit to https://github.com/tgautam03/CUDA-C/blob/master/05_tiled_mat_mul/tiled_mat_mul_gpu.cu
	/// </summary>
	/// <param name="status"></param>
	void checkCuda(const cudaError_t &status) const {
		if (status != cudaSuccess) {
			std::cout << cudaGetErrorString(status) << "in " << __FILE__  << " at line " << __LINE__ << std::endl;
			exit(-1);
		}
	}
};
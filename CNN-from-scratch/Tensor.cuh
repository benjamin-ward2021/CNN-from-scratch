#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <string>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "TensorKernels.cuh"

using std::vector, std::cout, std::endl, std::string, std::min, std::max, std::default_random_engine, std::normal_distribution, std::uniform_real_distribution, std::shuffle;

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
	Tensor(const vector<int> &dims) : dims(dims) {
		assert(dims.size() > 0);
		int size = createCoordinateConversionLookupTable(dims);
		data = vector<T>(size);
	}

	/// <summary>
	/// Initializes data using the given vector.
	/// </summary>
	/// <param name="dims"></param>
	Tensor(const vector<T> &data, const vector<int> &dims) : data(data), dims(dims) {
		assert(dims.size() > 0);
		int size = createCoordinateConversionLookupTable(dims);

		// Verify that the data passed in matches the expected size
		assert(data.size() == size);
	}

	bool operator==(const Tensor<T> &other) const {
		return data == other.data && dims == other.dims && coordinateConversionLookupTable == other.coordinateConversionLookupTable;
	}

	bool operator!=(const Tensor<T> &other) const {
		return !(*this == other);
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
	T get(const vector<int> &indices) const {
		int flattenedIndex = getFlattenedIndex(indices);
		return data[flattenedIndex];
	}

	/// <summary>
	/// Sets the element to value. Excluded indices are assumed to be zero.
	/// </summary>
	/// <param name="indices"></param>
	/// <param name="value"></param>
	void set(const vector<int> &indices, T value) {
		int flattenedIndex = getFlattenedIndex(indices);
		data[flattenedIndex] = value;
	}

	/// <summary>
	/// Sets all elements in the tensor to a random value using the normal distribution and random engine.
	/// Parameters are passed by mutable reference and can be changed.
	/// </summary>
	/// <param name="normalDistribution"></param>
	/// <param name="randomEngine"></param>
	void setToRandom(normal_distribution<T> normalDistribution, default_random_engine randomEngine) {
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
	void setToRandom(uniform_real_distribution<T> &uniformDistribution, default_random_engine &randomEngine) {
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
	vector<int> getDims() const {
		return dims;
	}

	/// <summary>
	/// Reshapes the tensor.
	/// </summary>
	/// <param name="dims"></param>
	void reinterpretDims(const vector<int> &dims) {
		int size = createCoordinateConversionLookupTable(dims);
		assert(size == data.size());
		this->dims = dims;
	}

	/// <summary>
	/// Flattens one dimension of the tensor.
	/// (Ex. { 2,3,4,5 } becomes { 2*3,4,5 } = { 6,4,5 }).
	/// </summary>
	void flattenOnce() {
		assert(dims.size() > 1);
		vector<int> newDims = dims;
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
	/// Provides a way to get batchSize random indices of samples. 
	/// Used in combination with getBatch, so inputs and targets can share the same mapping.
	/// </summary>
	/// <param name="batchSize"></param>
	/// <param name="randomEngine"></param>
	/// <returns></returns>
	[[nodiscard]]
	vector<int> getRandomIndices(int batchSize, default_random_engine &randomEngine) const {
		const int totalNumSamples = dims[0];
		assert(batchSize <= totalNumSamples);
		vector<int> indices(totalNumSamples);
		for (int i = 0; i < totalNumSamples; i++) {
			indices[i] = i;
		}

		shuffle(indices.begin(), indices.end(), randomEngine);

		return vector<int>(indices.begin(), indices.begin() + batchSize);
	}

	/// <summary>
	/// Gets a batch of samples based on their indices.
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>A tensor with dims = { batchSize,dims[1],dims[2],dims[3]... }</returns>
	[[nodiscard]]
	Tensor<T> getBatch(const vector<int> &indices) const {
		const int batchSize = static_cast<int>(indices.size());
		const int totalNumSamples = dims[0];
		assert(batchSize <= totalNumSamples);
		vector<int> batchDims = dims;
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
			int i_max = min(i + tileSize, M);
			for (int j = 0; j < N; j += tileSize) {
				int j_max = min(j + tileSize, N);
				for (int k = 0; k < K; k += tileSize) {
					int k_max = min(k + tileSize, K);
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
			int i_max = min(i + tileSize, rows);
			for (int j = 0; j < cols; j += tileSize) {
				int j_max = min(j + tileSize, cols);
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

	/// <summary>
	/// Sums over an axis in the tensor. 
	/// </summary>
	/// <param name="dim">The index of the dimension, starting at 0.</param>
	/// <param name="keepDummyDim">Whether to keep the dimension as 1, or remove it entirely.</param>
	/// <returns>New tensor with values summed along the dimension</returns>
	[[nodiscard]]
	Tensor<T> sum(int dim, bool keepDummyDim = false) const {
		assert(dim < dims.size());

		vector<int> sumDims = dims;
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
		int higherDimsSize = data.size() / (dims[dim] * coordinateConversionLookupTable[dim]);
		assert(higherDimsSize > 0);

		// Size of target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then currentDimsSize = 5
		int currentDimsSize = dims[dim];

		// Number of elements in dimensions after the target dimension
		// (Ex. if dims = { 3,7,5,2,4 } and dim = 2, then lowerDimsSize = 2 * 4 = 8
		int lowerDimsSize = coordinateConversionLookupTable[dim];
		assert(lowerDimsSize > 0);

		for (int higherIndex = 0; higherIndex < higherDimsSize; higherIndex++) {
			for (int lowerIndex = 0; lowerIndex < lowerDimsSize; lowerIndex++) {
				T accumulator = static_cast<T>(0);
				for (int currentIndex = 0; currentIndex < currentDimsSize; currentIndex++) {
					int dataIndex = 
						higherIndex * (currentDimsSize * lowerDimsSize) +
						currentIndex * lowerDimsSize +
						lowerIndex;

					accumulator += data[dataIndex];
				}

				int resultDataIndex = 
					higherIndex * lowerDimsSize +
					lowerIndex;

				result.data[resultDataIndex] = accumulator;
			}
		}

		return result;
	}

	/// <summary>
	/// Adds two tensors of the same shape together. Must specify return type as first template type.
	/// </summary>
	/// <typeparam name="U">Return type</typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of sums.</returns>
	template <typename U, typename V>
	[[nodiscard]]
	Tensor<U> elementwiseAdd(const Tensor<V> &other) const {
		assert(dims == other.dims);
		vector<U> sum(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			sum[i] = static_cast<U>(data[i] + other.data[i]);
		}
	
		return Tensor<U>(sum, dims);
	}

	/// <summary>
	/// Adds two tensors of the same shape, in place.
	/// (I.e. data += other.data).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	template <typename U>
	void elementwiseAddInPlace(const Tensor<U> &other) {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			data[i] += other.data[i];
		}
	}

	/// <summary>
	/// Adds two tensors together, where if other has fewer dimensions than this, it gets broadcasted up.
	/// Must specify return type as first template type.
	/// Note that this doesn't actually check that the dimensions are compatible; it only asserts that the data sizes are compatible.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of sums.</returns>
	template <typename U, typename V>
	[[nodiscard]]
	Tensor<U> broadcastAdd(const Tensor<V> &other) const {
		assert(dims >= other.dims && data.size() >= other.data.size() && data.size() % other.data.size() == 0);
		vector<U> sum(data.size());
		int broadcastFactor = data.size() / other.data.size();
		for (int i = 0; i < broadcastFactor; i++) {
			for (int j = 0; j < other.data.size(); j++) {
				int dataIndex = i * other.data.size() + j;
				sum[dataIndex] = static_cast<U>(data[dataIndex] + other.data[j]);
			}
		}

		return Tensor<U>(sum, dims);
	}

	/// <summary>
	/// Adds two tensors together in place, where if other has fewer dimensions than this, it gets broadcasted up.
	/// See https://www.geeksforgeeks.org/tensor-broadcasting/
	/// Note that this doesn't actually check that the dimensions are compatible; it only asserts that the data sizes are compatible.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of sums.</returns>
	template <typename U>
	void broadcastAddInPlace(const Tensor<U> &other) {
		assert(dims.size() >= other.dims.size() && data.size() >= other.data.size() && data.size() % other.data.size() == 0);
		int broadcastFactor = static_cast<int>(data.size() / other.data.size());
		for (int i = 0; i < broadcastFactor; i++) {
			for (int j = 0; j < other.data.size(); j++) {
				int dataIndex = static_cast<int>(i * other.data.size() + j);
				data[dataIndex] += other.data[j];
			}
		}
	}

	/// <summary>
	/// Subtracts two tensors of the same shape. Must specify return type as first template type.
	/// </summary>
	/// <typeparam name="U">Return type</typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of differences.</returns>
	template <typename U, typename V>
	[[nodiscard]]
	Tensor<U> elementwiseSubtract(const Tensor<V> &other) const {
		assert(dims == other.dims);
		vector<U> difference(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			difference[i] = static_cast<U>(data[i] - other.data[i]);
		}

		return Tensor<U>(difference, dims);
	}

	/// <summary>
	/// Subtracts two tensors of the same shape, in place.
	/// (I.e. data -= other.data).
	/// To store the result in other, see elementwiseSubtractInReversePlace.
	/// (I.e. other.data = data - other.data).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	template <typename U>
	void elementwiseSubtractInPlace(const Tensor<U> &other) {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			data[i] -= other.data[i];
		}
	}

	/// <summary>
	/// Subtracts two tensors in place, where if other has fewer dimensions than this, it gets broadcasted up.
	/// See https://www.geeksforgeeks.org/tensor-broadcasting/
	/// Note that this doesn't actually check that the dimensions are compatible; it only asserts that the data sizes are compatible.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of differences.</returns>
	template <typename U>
	void broadcastSubtractInPlace(const Tensor<U> &other) {
		assert(dims.size() >= other.dims.size() && data.size() >= other.data.size() && data.size() % other.data.size() == 0);
		int broadcastFactor = static_cast<int>(data.size() / other.data.size());
		for (int i = 0; i < broadcastFactor; i++) {
			for (int j = 0; j < other.data.size(); j++) {
				int dataIndex = static_cast<int>(i * other.data.size() + j);
				data[dataIndex] -= other.data[j];
			}
		}
	}

	/// <summary>
	/// Subtracts two tensors of the same shape, in place, storing the result in other.
	/// (I.e. other.data = data - other.data).
	/// To store the result in this tensor, see elementwiseSubtractInPlace.
	/// (I.e. data -= other.data).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	template <typename U>
	void elementwiseSubtractInReversePlace(Tensor<U> &other) const {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			other.data[i] = data[i] - other.data[i];
		}
	}

	/// <summary>
	/// Multiplies two tensors of the same shape, in an elementwise way (NOT matrix multiplication). Must specify return type as first template type.
	/// </summary>
	/// <typeparam name="U">Return type</typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of products.</returns>
	template <typename U, typename V>
	[[nodiscard]]
	Tensor<U> elementwiseMultiply(const Tensor<V> &other) const {
		assert(dims == other.dims);
		vector<U> product(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			product[i] = static_cast<U>(data[i] * other.data[i]);
		}

		return Tensor<U>(product, dims);
	}

	/// <summary>
	/// Multiplies two tensors of the same shape, in place.
	/// (I.e. data *= other.data).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	template <typename U>
	void elementwiseMultiplyInPlace(const Tensor<U> &other) {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			data[i] *= other.data[i];
		}
	}

	/// <summary>
	/// Multiplies a tensor with a scalar.
	/// </summary>
	/// <param name="multiplier"></param>
	/// <returns>The tensor of products.</returns>
	[[nodiscard]]
	Tensor<T> scalarMultiply(T multiplier) {
		Tensor<T> product(dims);
		for (int i = 0; i < data.size(); i++) {
			product.data[i] = data[i] * multiplier;
		}

		return product;
	}

	/// <summary>
	/// Divides two tensors of the same shape, in an elementwise way. Must specify return type as first template type.
	/// </summary>
	/// <typeparam name="U">Return type</typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of quotients.</returns>
	template <typename U, typename V>
	[[nodiscard]]
	Tensor<U> elementwiseDivide(const Tensor<V> &other) const {
		assert(dims == other.dims);
		vector<U> quotient(data.size());
		for (int i = 0; i < data.size(); i++) {
			// Only cast AFTER the operation has completed
			quotient[i] = static_cast<U>(data[i] / other.data[i]);
		}

		return Tensor<U>(quotient, dims);
	}

	/// <summary>
	/// Divides two tensors of the same shape, in place.
	/// (I.e. data /= other.data).
	/// To store the result in other, see elementwiseDivideInReversePlace.
	/// (I.e. other.data = data / other.data).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	template <typename U>
	void elementwiseDivideInPlace(const Tensor<U> &other) {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			data[i] /= other.data[i];
		}
	}

	/// <summary>
	/// Divides two tensors of the same shape, in place, storing the result in other.
	/// (I.e. other.data = data / other.data).
	/// To store the result in this tensor, see elementwiseDivideInPlace.
	/// (I.e. data /= other.data).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	template <typename U>
	void elementwiseDivideInReversePlace(Tensor<U> &other) const {
		assert(dims == other.dims);
		for (int i = 0; i < data.size(); i++) {
			other.data[i] = data[i] / other.data[i];
		}
	}
	
	/// <summary>
	/// </summary>
	/// <returns>A tensor where data is clamped to 0 if it is negative.</returns>
	[[nodiscard]]
	Tensor<T> relu() const {
		Tensor<T> ret(dims);
		for (int i = 0; i < data.size(); i++) {
			ret.data[i] = max(static_cast<T>(0), data[i]);
		}
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
	}

	/// <summary>
	/// Prints a 2d tensor to console. Dimensions are represented as { row,col }.
	/// </summary>
	void print2d(string delimiter = " ") const {
		assert(dims.size() == 2);
		for (int row = 0; row < dims[0]; row++) {
			for (int col = 0; col < dims[1]; col++) {
				cout << get({ row,col }) << delimiter;
			}
			cout << endl;
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
					cout << "#";
				}

				else {
					cout << " ";
				}
			}
			cout << endl;
		}
	}

	/// <summary>
	/// Prints a 3d tensor to console. Dimensions are represented as { row,col,depth }.
	/// </summary>
	void print3d(string delimiter = " ") const {
		assert(dims.size() == 3);
		for (int depth = 0; depth < dims[2]; depth++) {
			cout << "Depth: " << depth << endl;
			for (int row = 0; row < dims[0]; row++) {
				for (int col = 0; col < dims[1]; col++) {
					cout << get({ row,col,depth}) << delimiter;
				}
				cout << endl;
			}
		}
	}

private:
	/// <summary>
	/// A vector containing the flattened data from a multidimensional tensor.
	/// </summary>
	vector<T> data;
	/// <summary>
	/// The dimensions of the tensor. 
	/// (Ex. { 2,3 } for a matrix with 2 rows and 3 columns).
	/// </summary>
	vector<int> dims;
	/// <summary>
	/// Lookup table used for converting multidimensional coordinates to a flattened representation.
	/// </summary>
	vector<int> coordinateConversionLookupTable;

	/// <summary>
	/// Creates the coordinate conversion table, and returns the total size of data.
	/// </summary>
	/// <param name="dims"></param>
	/// <returns>The total size of the data vector.</returns>
	int createCoordinateConversionLookupTable(const vector<int> &dims) {
		coordinateConversionLookupTable = vector<int>(dims.size());

		// Total number of elements in data
		int size = 1;
		for (int i = 0; i < dims.size(); i++) {
			// Initialize back to front since we can use size for the lookup table if we do it like this
			int currentIndex = static_cast<int>(dims.size()) - 1 - i;
			coordinateConversionLookupTable[currentIndex] = size;
			size *= dims[currentIndex];
		}

		return size;
	}

	/// <summary>
	/// Converts a multidimensional set of indices to the equivalent flattened index
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The flattened index</returns>
	[[nodiscard]]
	int getFlattenedIndex(const vector<int> &indices) const {
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
			cout << cudaGetErrorString(status) << "in " << __FILE__  << " at line " << __LINE__ << endl;
			exit(-1);
		}
	}
};
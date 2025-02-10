#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <string>

using std::vector, std::cout, std::endl, std::string;

/// <summary>
/// A multidimensional array.
/// </summary>
/// <typeparam name="T"></typeparam>
template<typename T>
class Tensor {
public:
	// All types of tensors are friends with all other types of tensors, and can access their private variables.
	// (Ex. Tensor<float> can access dims of a Tensor<int>).
	friend class Tensor;

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
	/// Initializes data with the given vector.
	/// (Ex. 0 for int).
	/// </summary>
	/// <param name="dims"></param>
	Tensor(const vector<T> &data, const vector<int> &dims) : data(data), dims(dims) {
		assert(dims.size() > 0);
		int size = createCoordinateConversionLookupTable(dims);

		// Verify that the data passed in matches the expected size
		assert(data.size() == size);
	}

	/// <summary>
	/// Gets an element from data. Excluded indices are assumed to be zero.
	/// (Ex. if indices = { 2,3 }, get the element at row 2 column 3. Equivalent to { 2,3,0 } for a 3d tensor).
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The element at the specified indices.</returns>
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
	/// Reshapes the tensor.
	/// </summary>
	/// <param name="dims"></param>
	void reinterpretDims(const vector<int> &dims) {
		// This looks ugly but we don't want to make the function call in release mode, so we put it in the assert.
		assert(data.size() == createCoordinateConversionLookupTable(dims));
		this->dims = dims;
	}

	/// <summary>
	/// Flattens one dimension of the tensor.
	/// (Ex. { 2,3,4 } becomes { 6,4 }).
	/// </summary>
	void flattenOnce() {
		assert(dims.size() > 1);
		vector<int> newDims = dims;
		newDims[0] *= newDims[1];
		newDims.erase(newDims.begin() + 1);
		reinterpretDims(newDims);
	}

	/// <summary>
	/// Multiplies two matrices, specifically this * other. They don't have to be the same type, and neither does the return type.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns></returns>
	template <typename U, typename V>
	Tensor<U> matrixMultiply(const Tensor<V> &other) const {
		assert(dims.size() == 2 && other.dims.size() == 2);
		assert(dims[1] == other.dims[0]);

		Tensor<U> product({ dims[0],other.dims[1] });
		for (int i = 0; i < dims[0]; i++) {
			for (int j = 0; j < other.dims[1]; j++) {
				U value = static_cast<U>(0);
				for (int k = 0; k < other.dims[0]; k++) {
					value += static_cast<U>(get({ i,k }) * other.get({ k,j }));
				}

				product.set({ i,j }, value);
			}
		}

		return product;
	}

	/// <summary>
	/// Adds two tensors of the same shape together.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of sums.</returns>
	template <typename U>
	Tensor<T> elementwiseAdd(const Tensor<U> &other) const {
		assert(dims == other.dims);
		vector<T> sum;
		sum.reserve(data.size());
		for (int i = 0; i < data.size(); i++) {
			sum = data[i] + other.data[i];
		}

		return Tensor<T>(sum, dims);
	}

	/// <summary>
	/// Subtracts two tensors of the same shape.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of differences.</returns>
	template <typename U>
	Tensor<T> elementwiseSubtract(const Tensor<U> &other) const {
		assert(dims == other.dims);
		vector<T> difference;
		difference.reserve(data.size());
		for (int i = 0; i < data.size(); i++) {
			difference = data[i] - other.data[i];
		}

		return Tensor<T>(difference, dims);
	}

	/// <summary>
	/// Multiplies two tensors of the same shape, in an elementwise way (NOT matrix multiplication).
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of products.</returns>
	template <typename U>
	Tensor<T> elementwiseMultiply(const Tensor<U> &other) const {
		assert(dims == other.dims);
		vector<T> product;
		product.reserve(data.size());
		for (int i = 0; i < data.size(); i++) {
			product = data[i] * other.data[i];
		}

		return Tensor<T>(product, dims);
	}

	/// <summary>
	/// Divides two tensors of the same shape, in an elementwise way.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <param name="other"></param>
	/// <returns>The tensor of quotients.</returns>
	template <typename U>
	Tensor<T> elementwiseDivide(const Tensor<U> &other) const {
		assert(dims == other.dims);
		vector<T> quotient;
		quotient.reserve(data.size());
		for (int i = 0; i < data.size(); i++) {
			quotient = data[i] / other.data[i];
		}

		return Tensor<T>(quotient, dims);
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
	int getFlattenedIndex(const vector<int> &indices) const {
		int flattenedIndex = 0;
		for (int i = 0; i < indices.size(); i++) {
			// Make sure the index isn't out of bounds
			assert(indices[i] < dims[i]);
			flattenedIndex += coordinateConversionLookupTable[i] * indices[i];
		}

		return flattenedIndex;
	}
};
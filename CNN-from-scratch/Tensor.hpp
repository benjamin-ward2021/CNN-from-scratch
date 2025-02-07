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
	Tensor(vector<int> dims) {
		assert(dims.size() > 0);
		this->dims = dims;
		coordinateConversionLookupTable = vector<int>(dims.size());

		// Total number of elements in data
		int size = 1;
		for (int i = 0; i < dims.size(); i++) {
			// Initialize back to front since we can use size for the lookup table if we do it like this
			int currentIndex = static_cast<int>(dims.size()) - 1 - i;
			coordinateConversionLookupTable[currentIndex] = size;
			size *= dims[currentIndex];
		}

		data = vector<T>(size);
	}

	/// <summary>
	/// Initializes data with the given vector.
	/// (Ex. 0 for int).
	/// </summary>
	/// <param name="dims"></param>
	Tensor(vector<T> data, vector<int> dims) {
		assert(dims.size() > 0);
		this->dims = dims;
		coordinateConversionLookupTable = vector<int>(dims.size());

		// Total number of elements in data
		int size = 1;
		for (int i = 0; i < static_cast<int>(dims.size()); i++) {
			// Initialize back to front since we can use size for the lookup table if we do it like this
			int currentIndex = static_cast<int>(dims.size()) - 1 - i;
			coordinateConversionLookupTable[currentIndex] = size;
			size *= dims[currentIndex];
		}

		// Verify that the data passed in matches the expected size
		assert(data.size() == size);
		this->data = data;
	}

	/// <summary>
	/// Gets an element from data. Excluded indices are assumed to be zero.
	/// (Ex. if indices = { 2,3 }, get the element at row 2 column 3. Equivalent to { 2,3,0 } for a 3d tensor).
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The element at the specified indices.</returns>
	T get(const vector<int> &indices) {
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
	/// Multiplies two matrices, specifically this * other. They don't have to be the same type, and neither does the return type.
	/// </summary>
	/// <typeparam name="U"></typeparam>
	/// <typeparam name="V"></typeparam>
	/// <param name="other"></param>
	/// <returns></returns>
	template <typename U, typename V>
	Tensor<U> matrixMultiply(Tensor<V> other) {
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
	/// Prints a 2d tensor to console. Dimensions are represented as { row,col }.
	/// </summary>
	void print2d(string delimiter = " ") {
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
	void print2dThreshold(T threshold) {
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
	void print3d(string delimiter = " ") {
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
	/// (Ex. { 2,3 } for a matrix with 2 rows and 3 columns)
	/// </summary>
	vector<int> dims;
	/// <summary>
	/// Lookup table used for converting multidimensional coordinates to a flattened representation.
	/// </summary>
	vector<int> coordinateConversionLookupTable;

	/// <summary>
	/// Converts a multidimensional set of indices to the equivalent flattened index
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The flattened index</returns>
	int getFlattenedIndex(const vector<int> &indices) {
		int flattenedIndex = 0;
		for (int i = 0; i < indices.size(); i++) {
			// Make sure the index isn't out of bounds
			assert(indices[i] < dims[i]);
			flattenedIndex += coordinateConversionLookupTable[i] * indices[i];
		}

		return flattenedIndex;
	}
};
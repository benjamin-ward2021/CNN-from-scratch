#pragma once
#include <vector>
#include <cassert>
#include <iostream>

using std::vector, std::cout, std::endl;

template<typename T>
class Tensor {
public:
	/// <summary>
	/// Initializes data with size of sum(dims), where each element is the types default value.
	/// (Ex. 0 for int).
	/// </summary>
	/// <param name="dims"></param>
	Tensor(vector<size_t> dims) {
		assert(dims.size() > 0);
		this->dims = dims;
		coordinateConversionLookupTable = vector<size_t>(dims.size());

		// Total number of elements in data
		size_t size = 1;
		for (int i = dims.size() - 1; i >= 0; i--) {
			coordinateConversionLookupTable[i] = size;
			size *= dims[i];
		}

		data = vector<T>(size);
	}

	/// <summary>
	/// Gets an element from data. Excluded indices are assumed to be zero.
	/// (Ex. if indices = { 2,3 }, get the element at row 2 column 3. Equivalent to { 2,3,0 } for a 3d tensor).
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The element at the specified indices.</returns>
	T get(const vector<size_t> &indices) {
		size_t flattenedIndex = getFlattenedIndex(indices);
		return data[flattenedIndex];
	}

	/// <summary>
	/// Sets the element to value. Excluded indices are assumed to be zero.
	/// </summary>
	/// <param name="indices"></param>
	/// <param name="value"></param>
	void set(const vector<size_t> &indices, T value) {
		size_t flattenedIndex = getFlattenedIndex(indices);
		data[flattenedIndex] = value;
	}

	/// <summary>
	/// Prints a 2d tensor to console
	/// </summary>
	void print2d() {
		assert(dims.size() == 2);
		for (size_t row = 0; row < dims[0]; row++) {
			for (size_t col = 0; col < dims[1]; col++) {
				cout << get({ row,col }) << " ";
			}
			cout << endl;
		}
	}

	/// <summary>
	/// Prints a 3d tensor to console
	/// </summary>
	void print3d() {
		assert(dims.size() == 3);
		for (size_t depth = 0; depth < dims[2]; depth++) {
			cout << "Depth: " << depth << endl;
			for (size_t row = 0; row < dims[0]; row++) {
				for (size_t col = 0; col < dims[1]; col++) {
					cout << get({ row,col,depth}) << " ";
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
	vector<size_t> dims;
	/// <summary>
	/// Lookup table used for converting multidimensional coordinates to a flattened representation.
	/// </summary>
	vector<size_t> coordinateConversionLookupTable;

	/// <summary>
	/// Converts a multidimensional set of indices to the equivalent flattened index
	/// </summary>
	/// <param name="indices"></param>
	/// <returns>The flattened index</returns>
	size_t getFlattenedIndex(const vector<size_t> &indices) {
		size_t flattenedIndex = 0;
		for (size_t i = 0; i < indices.size(); i++) {
			// Make sure the index isn't out of bounds
			assert(indices[i] < dims[i]);
			flattenedIndex += coordinateConversionLookupTable[i] * indices[i];
		}

		return flattenedIndex;
	}
};
#pragma once
#include <vector>

using std::vector;

template<typename T>
class Tensor {
private:
	/// <summary>
	/// A vector containing the flattened data from a multidimensional tensor.
	/// </summary>
	vector<T> data;
public:
	/// <summary>
	/// Initializes data with size of sum(dims), where each element is the types default value.
	/// (ex. 0 for int)
	/// </summary>
	/// <param name="dims"></param>
	Tensor(const vector<int> &dims) {
		// Total number of elements in data
		size_t size = 0;
		for (int i = 0; i < dims.size(); i++) {
			size += dims[i];
		}

		data = vector<T>(size);
		for (int i = 0; i < data.size(); i++) {
			std::cout << data[i] << std::endl;
		}
	}
};
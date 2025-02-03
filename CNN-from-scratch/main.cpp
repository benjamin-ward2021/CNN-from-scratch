#include <iostream>
#include <vector>
#include "Tensor.hpp"

using std::vector, std::cout, std::endl;

int main() {
	vector<int> dims = { 1000,2,3,1 };
	Tensor<double> test(dims);
}
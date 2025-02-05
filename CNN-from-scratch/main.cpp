#include <iostream>
#include <vector>
#include "Tensor.hpp"

using std::vector, std::cout, std::endl;

int main() {
	vector<size_t> dims = { 7,3,3 };
	Tensor<double> test(dims);
	test.set({ 1,1 }, 2);
	test.set({ 1,2 }, 3);
	test.set({ 0,2 }, 4);
	test.set({ 5,0,1 }, 5.1);
	test.print3d();
}
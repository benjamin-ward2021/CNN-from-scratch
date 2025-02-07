#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "Tensor.hpp"
#include "MNISTLoader.hpp"

using std::vector, std::cout, std::endl, std::string;

//void testTensorSpeed();

int main() {
	MNISTLoader trainData("MNIST Data\\train-images.idx3-ubyte", "MNIST Data\\train-labels.idx1-ubyte", 100); // TODO: Change to 60000
	trainData.printData(0, 10);
	Tensor<double> t1({ 2,3 });
	t1.set({ 0,0 }, 0);
	t1.set({ 0,1 }, 1);
	t1.set({ 0,2 }, 2.1);
	t1.set({ 1,0 }, 1.5);
	t1.set({ 1,1 }, 3);
	t1.set({ 1,2 }, 5.25);

	Tensor<float> t2({ 3,1 });
	t2.set({ 0,0 }, 1.5);
	t2.set({ 1,0 }, 3);
	t2.set({ 2,0 }, 5.5);

	Tensor<double> t3 = t1.matrixMultiply<double>(t2);
	// [14.55; 40.125] expected
	t3.print2d();
	//MNISTLoader testData("MNIST Data\\t10k-images.idx3-ubyte", "MNIST Data\\t10k-labels.idx1-ubyte", 10000);
}

// This is just a function for timing execution speeds. Used only for debugging.
//void testTensorSpeed() {
//    using std::chrono::high_resolution_clock;
//    using std::chrono::duration;
//
//    Tensor<double> toTest({ 10,10,10 });
//
//    auto t1 = high_resolution_clock::now();
//    for (int i = 0; i < 1000000; i++) {
//        toTest.set({ 2,2,2 }, 1.1);
//    }
//    auto t2 = high_resolution_clock::now();
//
//    /* Getting number of milliseconds as a double. */
//    duration<double, std::milli> ms_double = t2 - t1;
//    std::cout << ms_double.count() << "ms\n";
//}
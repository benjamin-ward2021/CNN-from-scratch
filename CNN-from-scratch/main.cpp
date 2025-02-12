#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "Tensor.hpp"
#include "MNISTLoader.hpp"
#include "Layer.hpp"
#include "FullyConnected.hpp"

using std::vector, std::cout, std::endl, std::string;

void measureTensorSpeed();
void testMatrixMultiplication();
void testMatrixMultiplication2();
void testElementwiseAdd();
void testBroadcastAdd();
void testFullyConnectedForward();

int main() {
	testFullyConnectedForward();
	//MNISTLoader trainData("MNIST Data\\train-images.idx3-ubyte", "MNIST Data\\train-labels.idx1-ubyte", 100); // TODO: Change to 60000
	//trainData.printData(0, 10);
	//testMatrixMultiplication();

	//FullyConnected<double> fc(5, 10, 0, Layer::WeightInitializationHeuristic::heNormal);
	//MNISTLoader testData("MNIST Data\\t10k-images.idx3-ubyte", "MNIST Data\\t10k-labels.idx1-ubyte", 10000);
}

// This is just a function for timing execution speeds. Used only for debugging.
void measureTensorSpeed() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    Tensor<double> toTest({ 10,10,10 });

    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
        toTest.set({ 2,2,2 }, 1.1);
    }
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
}

void testMatrixMultiplication() {
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
	assert(t3.get({ 0 }) == 14.55 && t3.get({ 1 }) == 40.125);
}

void testMatrixMultiplication2() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 32,10000 });
	t1.set({ 0,0 }, 0);
	t1.set({ 0,1 }, 1);
	t1.set({ 0,2 }, 2.1);
	t1.set({ 1,0 }, 1.5);
	t1.set({ 1,1 }, 3);
	t1.set({ 1,2 }, 5.25);

	Tensor<double> t2({ 10000,3000 });
	t2.set({ 0,0 }, 1.5);
	t2.set({ 1,0 }, 3);
	t2.set({ 2,0 }, 5.5);
	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 10; i++) {
		Tensor<double> t3 = t1.matrixMultiply<double>(t2);
		//Tensor<double> t3 = t1.matrixMultiply<double>(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void testElementwiseAdd() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 2,3,10,10 });
	t1.set({ 0,0 }, 3.2);
	t1.set({ 0,1 }, 1);
	t1.set({ 0,2 }, 2.1);
	t1.set({ 1,0 }, 1.5);
	t1.set({ 1,1 }, 3);
	t1.set({ 1,2 }, 5.25);

	Tensor<float> t2({ 2,3,10,10 });
	t2.set({ 0,0 }, 1.5);
	t2.set({ 0,1 }, 3.5);
	t2.set({ 0,2 }, 4.0);
	t2.set({ 1,0 }, 4.5);
	t2.set({ 1,1 }, 10.5);
	t2.set({ 1,2 }, 15.1);

	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 100000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		t1.elementwiseMultiplyInPlace(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
	//assert(t3.get({ 0 }) == 14.55 && t3.get({ 1 }) == 40.125);
}

void testBroadcastAdd() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 2,3 });
	t1.set({ 0,0 }, 3.2);
	t1.set({ 0,1 }, 1);
	t1.set({ 0,2 }, 2.1);
	t1.set({ 1,0 }, 1.5);
	t1.set({ 1,1 }, 3);
	t1.set({ 1,2 }, 5.25);

	Tensor<float> t2({ 3 });
	t2.set({ 0 }, 1.5);
	t2.set({ 1 }, 3.5);
	t2.set({ 2 }, 4.0);

	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 100000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		t1.broadcastAddInPlace(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
	//assert(t3.get({ 0 }) == 14.55 && t3.get({ 1 }) == 40.125);
}

void testFullyConnectedForward() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;

	FullyConnected<double> fc(2, 1);

	Tensor<double> t2({ 1,2 });
	t2.set({ 0,0 }, 2);
	t2.set({ 0,1 }, 3);

	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 100000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		auto t3 = fc.forward(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}
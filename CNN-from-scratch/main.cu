#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>

#include "Tensor.cuh"
#include "MNISTLoader.cuh"
#include "Layer.cuh"
#include "FullyConnected.cuh"
#include "ReLU.cuh"
#include "SoftmaxCrossEntropy.cuh"
#include "XORGenerator.cuh"
#include "Network.cuh"

using std::vector, std::cout, std::endl, std::string, std::unique_ptr;

void measureTensorSpeed();
void testMatrixMultiplication();
void testMatrixMultiplication2();
void testFullyConnectedForward();
void testFullyConnectedForward2();
void testTranspose();
void testMatrixRowSum();
void testMatrixColumnSum();
void testFullyConnectedBackward();
void testCuda();
void testMatrixMultiplicationGPU();
void compareCpuGpuMatrixMultiplication();
void testXORGenerator();
void testLayers();
void testSum();
void testNetwork();

int main() {
	// TODO: Add nodiscard / noexcept to certain functions
	testNetwork();
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

	Tensor<double> t2({ 3,1 });
	t2.set({ 0,0 }, 1.5);
	t2.set({ 1,0 }, 3);
	t2.set({ 2,0 }, 5.5);

	Tensor<double> t3 = t1.matrixMultiply(t2);
	// [14.55; 40.125] expected
	assert(t3.get({ 0 }) == 14.55 && t3.get({ 1 }) == 40.125);
}

void testMatrixMultiplication2() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 32,5000 });
	t1.set({ 0,0 }, 0);
	t1.set({ 0,1 }, 1);
	t1.set({ 0,2 }, 2.1);
	t1.set({ 1,0 }, 1.5);
	t1.set({ 1,1 }, 3);
	t1.set({ 1,2 }, 5.25);

	Tensor<double> t2({ 5000,2500 });
	t2.set({ 0,0 }, 1.5);
	t2.set({ 1,0 }, 3);
	t2.set({ 2,0 }, 5.5);
	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 10; i++) {
		Tensor<double> t3 = t1.matrixMultiply(t2);
		//Tensor<double> t3 = t1.matrixMultiply<double>(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

// TODO: Actually make a test case that asserts if the output is correct
void testFullyConnectedForward() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;

	FullyConnected<double> fc(2, 1, 0.01);

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

void testFullyConnectedForward2() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;

	FullyConnected<double> fc(5000, 5000, 0.001);

	Tensor<double> t2({ 32,5000 });
	Tensor<double> t3({ 32,5000 });
	std::uniform_real_distribution<double> uniform(0.0, static_cast<double>(sqrt(2.0 / 10)));
	std::default_random_engine engine(1);
	t2.setToRandom(uniform, engine);
	t3.setToRandom(uniform, engine);

	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 10; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		fc.forward(t2);
		fc.backward(t3);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}


void testTranspose() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 5000,2500 });
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
	for (int i = 0; i < 100; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		auto t3 = t1.transpose();
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void testMatrixRowSum() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 32,5000 });
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
	for (int i = 0; i < 1000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		auto t3 = t1.matrixRowSum();
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void testMatrixColumnSum() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 32,5000 });
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
	for (int i = 0; i < 1000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		auto t3 = t1.matrixColumnSum();
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void testFullyConnectedBackward() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;

	FullyConnected<double> fc(2, 1, 0,01);

	Tensor<double> t2({ 1,2 });
	t2.set({ 0,0 }, 2);
	t2.set({ 0,1 }, 3);

	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 100000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		auto t3 = fc.forward(t2);
		auto t4 = fc.backward(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void testCuda() {
	// Initialize arrays A, B, and C.
	//double A[3], B[3], C[3];

	// Populate arrays A and B.
	//A[0] = 5; A[1] = 8; A[2] = 3;
	//B[0] = 7; B[1] = 6; B[2] = 4;

	// Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA.
	//TensorMathGPU::kernel(A, B, C, 3);

	// Print out result.
//	std::cout << "C = " << C[0] << ", " << C[1] << ", " << C[2] << std::endl;
}

void testMatrixMultiplicationGPU() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 2,3 });
	t1.set({ 0,0 }, 0);
	t1.set({ 0,1 }, 1);
	t1.set({ 0,2 }, 2.1);
	t1.set({ 1,0 }, 1.5);
	t1.set({ 1,1 }, 3);
	t1.set({ 1,2 }, 5.25);

	Tensor<double> t2({ 3,1 });
	t2.set({ 0,0 }, 1.5);
	t2.set({ 1,0 }, 3);
	t2.set({ 2,0 }, 5.5);
	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 10; i++) {
		Tensor<double> t3 = t1.matrixMultiplyGPU(t2);
		//Tensor<double> t3 = t1.matrixMultiply<double>(t2);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void compareCpuGpuMatrixMultiplication() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	using std::uniform_real_distribution;
	using std::default_random_engine;

	uniform_real_distribution<double> dist(-1, 1);
	default_random_engine engine(0);

	Tensor<double> t1({ 32,5000 });
	t1.setToRandom(dist, engine);
	Tensor<double> t2({ 5000,5000 });
	t2.setToRandom(dist, engine);

	auto time1 = high_resolution_clock::now();
	for (int i = 0; i < 10; i++) {
		Tensor<double> t3 = t1.matrixMultiply(t2);
	}

	auto time2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << "CPU: " << ms_double.count() << "ms\n";

	auto time3 = high_resolution_clock::now();
	for (int i = 0; i < 10; i++) {
		Tensor<double> t3 = t1.matrixMultiplyGPU(t2);
	}

	auto time4 = high_resolution_clock::now();
	duration<double, std::milli> ms_double2 = time4 - time3;
	std::cout << "GPU: " << ms_double2.count() << "ms\n";
}

//void testXORGenerator() {
//	using std::default_random_engine;
//	default_random_engine randomEngine(0);
//
//	XORGenerator<double> generator(0);
//	generator.generate(64);
//	vector<int> indices = generator.getInputs().getRandomIndices(32, randomEngine);
//	Tensor<double> inputBatch = generator.getInputs().getBatch(indices);
//	Tensor<double> outputBatch = generator.getLabels().getBatch(indices);
//}

void testLayers() {
	FullyConnected<double> fc(100,100,0.01);
	ReLU<double> relu;
	SoftmaxCrossEntropy<double> softmax;
}

void testSum() {
	using std::chrono::high_resolution_clock;
	using std::chrono::duration;
	Tensor<double> t1({ 32,5000 });
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
	for (int i = 0; i < 1000; i++) {
		//Tensor<double> t3 = t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAdd<double>(t2);
		//t1.elementwiseAddInPlace(t2);
		auto t4 = t1.sum(0);
	}

	auto time2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = time2 - time1;
	std::cout << ms_double.count() << "ms\n";
}

void testNetwork() {
	XORGenerator<double> generator(0);
	generator.generate(1024);

	default_random_engine randomEngine(0);

	vector<unique_ptr<Layer<double>>> layers;
	layers.push_back(std::make_unique<FullyConnected<double>>(2, 11, 0.01));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<FullyConnected<double>>(11, 2, 0.01));
	layers.push_back(std::make_unique<SoftmaxCrossEntropy<double>>());

	Network network(std::move(layers));
	for (int i = 0; i < 10000; i++) {
		vector<int> indices = generator.getInputs().getRandomIndices(64, randomEngine);
		Tensor<double> inputs = generator.getInputs().getBatch(indices);
		Tensor<int> labels = generator.getLabels().getBatch(indices);

		Tensor<double> predicted = network.forward(inputs);
		if (i % 500 == 0) {
			cout << "Loss " << i << ": " << network.loss(predicted, labels) << endl;
		}
		network.backward(labels);
	}
}
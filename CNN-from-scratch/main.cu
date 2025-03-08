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
#include "Conv2D.cuh"

void testNetworkXOR();
void testNetworkMNIST();

int main() {
	// TODO: Add asserts to layers that don't have many. Add const to appropriate places.
	// add MaxPool a and BatchNorm layers.
	testNetworkMNIST();
}

void testNetworkXOR() {
	XORGenerator<double> generator(0);
	generator.generate(1024);

	std::default_random_engine randomEngine(1);

	std::vector<std::unique_ptr<Layer<double>>> layers;
	layers.push_back(std::make_unique<FullyConnected<double>>(5, 0.05, randomEngine()));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<FullyConnected<double>>(2, 0.05, randomEngine()));
	layers.push_back(std::make_unique<SoftmaxCrossEntropy<double>>());

	Network network(std::move(layers), {2});
	for (int i = 0; i < 1000000; i++) {
		std::vector<int> indices = generator.getInputs().getRandomIndices(64, randomEngine);
		Tensor<double> inputs = generator.getInputs().getBatch(indices);
		Tensor<int> labels = generator.getLabels().getBatch(indices);

		Tensor<double> predicted = network.forward(inputs);
		if (i % 500 == 0) {
			std::cout << "Loss " << i << ": " << network.loss(predicted, labels) << std::endl;
		}
		network.backward(labels);
	}

	Conv2D<double> test(0, 0, 0, 0, 12313);
}

void testNetworkMNIST() {
	constexpr double learningRate = 0.05;
	MNISTLoader<double> trainData("MNIST Data\\train-images.idx3-ubyte", "MNIST Data\\train-labels.idx1-ubyte", 60000);
	//trainData.printData(0, 10); TODO broken
	MNISTLoader<double> testData("MNIST Data\\t10k-images.idx3-ubyte", "MNIST Data\\t10k-labels.idx1-ubyte", 10000);

	std::default_random_engine randomEngine(1);

	std::vector<std::unique_ptr<Layer<double>>> layers;
	layers.push_back(std::make_unique<Conv2D<double>>(11, 3, learningRate, randomEngine(), 1, 1));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<Conv2D<double>>(33, 3, learningRate, randomEngine(), 2, 1));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<FullyConnected<double>>(11, learningRate, randomEngine()));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<FullyConnected<double>>(10, learningRate, randomEngine()));
	layers.push_back(std::make_unique<SoftmaxCrossEntropy<double>>());

	Network network(std::move(layers), { 1,28,28 });
	auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i <= 399; i++) {
		std::vector<int> indices = trainData.images.getRandomIndices(64, randomEngine);
		Tensor<double> inputs = trainData.images.getBatch(indices);
		Tensor<int> labels = trainData.labels.getBatch(indices);

		Tensor<double> predicted = network.forward(inputs);
		if (i % 10 == 0) {
			std::cout << "Loss " << i << ": " << network.loss(predicted, labels) << std::endl;
		}

		network.backward(labels);
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	auto durationInSeconds = std::chrono::duration<double, std::milli>(endTime - startTime).count() / 1000.0;

	Tensor<double> predicted = network.forward(testData.images);
	std::cout << "Test accuracy after " << durationInSeconds << " seconds: " << network.accuracy(predicted, testData.labels) << std::endl;
}
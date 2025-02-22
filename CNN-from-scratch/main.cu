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

void testNetwork();

int main() {
	// TODO: Add asserts to layers that don't have many
	testNetwork();
	//MNISTLoader<double> trainData("MNIST Data\\train-images.idx3-ubyte", "MNIST Data\\train-labels.idx1-ubyte", 60000); // TODO: Change to 60000
	//trainData.printData(0, 10);
	//MNISTLoader testData("MNIST Data\\t10k-images.idx3-ubyte", "MNIST Data\\t10k-labels.idx1-ubyte", 10000);
}

void testNetwork() {
	XORGenerator<double> generator(0);
	generator.generate(1024);

	std::default_random_engine randomEngine(1);

	std::vector<std::unique_ptr<Layer<double>>> layers;
	layers.push_back(std::make_unique<FullyConnected<double>>(2, 5, 0.05, randomEngine()));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<FullyConnected<double>>(5, 2, 0.05, randomEngine()));
	layers.push_back(std::make_unique<SoftmaxCrossEntropy<double>>());

	Network network(std::move(layers));
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
}
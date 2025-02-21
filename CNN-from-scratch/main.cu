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

void testNetwork() {
	XORGenerator<double> generator(0);
	generator.generate(1024);

	default_random_engine randomEngine(0);

	vector<unique_ptr<Layer<double>>> layers;
	layers.push_back(std::make_unique<FullyConnected<double>>(2, 11, 0.02));
	layers.push_back(std::make_unique<ReLU<double>>());
	layers.push_back(std::make_unique<FullyConnected<double>>(11, 2, 0.02));
	layers.push_back(std::make_unique<SoftmaxCrossEntropy<double>>());

	Network network(std::move(layers));
	for (int i = 0; i < 100000; i++) {
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
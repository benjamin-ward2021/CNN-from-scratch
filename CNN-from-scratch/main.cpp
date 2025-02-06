#include <iostream>
#include <vector>
#include <string>
#include "Tensor.hpp"
#include "MNISTLoader.hpp"

using std::vector, std::cout, std::endl, std::string;

int main() {
	MNISTLoader trainData("MNIST Data\\train-images.idx3-ubyte", "MNIST Data\\train-labels.idx1-ubyte", 60000);
	// Displays the first 20 images and their labels
	for (int i = 0; i < 20; i++) {
		trainData.images[i].print2dThreshold(0.5);
		cout << trainData.labels[i] << endl;
	}
	//MNISTLoader testData("MNIST Data\\t10k-images.idx3-ubyte", "MNIST Data\\t10k-labels.idx1-ubyte", 10000);
}
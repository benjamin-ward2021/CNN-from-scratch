#pragma once

#include <string>
#include <vector>

#include "Tensor.hpp"

using std::string, std::vector;

/// <summary>
/// Loads the MNIST dataset. 
/// Credit to https://github.com/arpaka/mnist-loader and https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
/// </summary>
class MNISTLoader {
public:
	MNISTLoader(string imagesPath, string labelsPath, int maxAmount);
	// Images will be grayscale, ranging from [0, 1] inclusive.
	vector<Tensor<double>> images;
	// Labels correspond with the handwritten digit. (Ex. image of handwritten 7 has label 7).
	vector<int> labels;
	void printData(int start, int end);
private:
	void loadImages(string path, int maxAmount);
	void loadLabels(string path, int maxAmount);
	int convertBytesToInt(char *bytes);
};
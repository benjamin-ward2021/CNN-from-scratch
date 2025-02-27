#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <concepts>

#include "Tensor.cuh"

/// <summary>
/// Loads the MNIST dataset. 
/// Credit to https://github.com/arpaka/mnist-loader and https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
/// </summary>
template <typename T> requires std::floating_point<T>
class MNISTLoader {
public:
	MNISTLoader(std::string imagesPath, std::string labelsPath, int maxAmount) {
		loadImages(imagesPath, maxAmount);
		loadLabels(labelsPath, maxAmount);
	}

	// Images will be grayscale, ranging from [0, 1] inclusive.
	Tensor<T> images;
	// Labels are one-hot encoded (Ex. { 0,0,1,0,0,0,0,0,0,0 } means the label is 2).
	Tensor<int> labels;

	/// <summary>
	/// Prints the specified images with their labels. Start is inclusive and end is exclusive.
	/// </summary>
	/// <param name="start"></param>
	/// <param name="end"></param>
	void printData(int start, int end) const {
		assert(start >= 0 && end <= images.getDims()[0]);
		for (int i = start; i < end; i++) {
			images.getSampleAt(i).print2dThreshold(0.5);
			std::cout << "Label: " << labels.convertLabelToInt(i) << std::endl;
		}
	}

private:
	/// <summary>
	/// Can be used to load either train or test images. 
	/// Currently, if maxAmount is less than the total amount, this will take the first maxAmount images.
	/// </summary>
	/// <param name="path"></param>
	/// <param name="maxAmount"></param>
	void loadImages(std::string path, int maxAmount) {
		std::ifstream ifs(path, std::ios::binary);
		char bytes[4];

		// The first 4 bytes are the "magic number", which is always 2051 for the images
		ifs.read(bytes, 4);
		assert(convertBytesToInt(bytes) == 2051);

		// The second 4 bytes are the number of images
		ifs.read(bytes, 4);
		int amount = convertBytesToInt(bytes);
		if (amount > maxAmount) {
			amount = maxAmount;
		}

		// The third 4 bytes are the number of rows
		ifs.read(bytes, 4);
		int rows = convertBytesToInt(bytes);

		// The fourth 4 bytes are the number of columns
		ifs.read(bytes, 4);
		int cols = convertBytesToInt(bytes);

		// Then the bytes for the images
		// Each pixel is one byte
		std::vector<char> imageBuffer(rows * cols);
		std::vector<T> imagesData(amount * rows * cols);
		for (int i = 0; i < amount; i++) {
			ifs.read(imageBuffer.data(), rows * cols);
			for (int j = 0; j < rows * cols; j++) {
				// Make sure we interpret the bits as unsigned
				imagesData[(rows * cols * i) + j] = static_cast<T>(static_cast<unsigned char>(imageBuffer[j]) / 255.0);
			}
		}

		images = Tensor<T>(imagesData, { amount,1,rows,cols });
	}

	/// <summary>
	/// Can be used to load either train or test labels.
	/// Currently, if maxAmount is less than the total amount, this will take the first maxAmount labels.
	/// </summary>
	/// <param name="path"></param>
	/// <param name="maxAmount"></param>
	void loadLabels(std::string path, int maxAmount) {
		std::ifstream ifs(path, std::ios::binary);
		char bytes[4];

		// The first 4 bytes are the "magic number", which is always 2049 for the labels
		ifs.read(bytes, 4);
		assert(convertBytesToInt(bytes) == 2049);

		// The second 4 bytes are the number of labels
		ifs.read(bytes, 4);
		int amount = convertBytesToInt(bytes);
		if (amount > maxAmount) {
			amount = maxAmount;
		}

		// Then the bytes for the labels
		// Each label is one byte
		char labelBuffer[1];
		// 10 since there are 10 digits and we are one-hot encoding the labels
		std::vector<int> labelsData(amount * 10);
		for (int i = 0; i < amount; i++) {
			ifs.read(labelBuffer, 1);
			assert(labelBuffer[0] < 10);
			for (int j = 0; j < 10; j++) {
				labelsData[(10 * i) + j] = labelBuffer[0] == j ? 1 : 0;
			}
		}

		labels = Tensor<int>(labelsData, { amount,10 });
	}

	/// <summary>
	/// Converts the next 4 bytes to an integer value.
	/// </summary>
	/// <param name="bytes"></param>
	/// <returns>The integer representation of the memory.</returns>
	[[nodiscard]]
	int convertBytesToInt(char *bytes) {
		return ((bytes[0] & 0xff) << 24) | ((bytes[1] & 0xff) << 16) |
			((bytes[2] & 0xff) << 8) | ((bytes[3] & 0xff) << 0);
	}
};
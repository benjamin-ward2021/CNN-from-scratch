#include "MNISTLoader.hpp"

#include <fstream>
#include <string>
#include <cassert>

#include "Tensor.hpp"

using std::string, std::ifstream, std::ios;

MNISTLoader::MNISTLoader(string imagesPath, string labelsPath, int maxAmount) {
	loadImages(imagesPath, maxAmount);
	loadLabels(labelsPath, maxAmount);
}

/// <summary>
/// Can be used to load either train or test images. 
/// Currently, if maxAmount is less than the total amount, this will take the first maxAmount images.
/// </summary>
/// <param name="path"></param>
/// <param name="maxAmount"></param>
void MNISTLoader::loadImages(string path, int maxAmount) {
	ifstream ifs(path, ios::binary);
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
	char *imageBuffer = new char[rows * cols];
	for (int i = 0; i < amount; i++) {
		ifs.read(imageBuffer, rows * cols);
		vector<double> image(rows * cols);
		for (int j = 0; j < rows * cols; j++) {
			// Make sure we interpret the bits as unsigned
			image[j] = static_cast<unsigned char>(imageBuffer[j]) / 255.0;
		}

		images.push_back(Tensor<double>(image, { rows,cols }));
	}

	delete[] imageBuffer;
	ifs.close();
}

/// <summary>
/// Can be used to load either train or test labels.
/// Currently, if maxAmount is less than the total amount, this will take the first maxAmount labels.
/// </summary>
/// <param name="path"></param>
/// <param name="maxAmount"></param>
void MNISTLoader::loadLabels(string path, int maxAmount) {
	ifstream ifs(path, ios::binary);
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
	for (int i = 0; i < amount; i++) {
		ifs.read(labelBuffer, 1);
		labels.push_back(labelBuffer[0]);
	}

	ifs.close();
}

/// <summary>
/// Converts the next 4 bytes to an integer value.
/// </summary>
/// <param name="bytes"></param>
/// <returns>The integer representation of the memory</returns>
int MNISTLoader::convertBytesToInt(char *bytes) {
	return ((bytes[0] & 0xff) << 24) | ((bytes[1] & 0xff) << 16) |
		   ((bytes[2] & 0xff) << 8)  | ((bytes[3] & 0xff) << 0);
}
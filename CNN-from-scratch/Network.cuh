#pragma once

#include "Layer.cuh"
#include "FullyConnected.cuh"
#include "ReLU.cuh"
#include "Softmax.cuh"
#include "Tensor.cuh"

using std::vector;

/// <summary>
/// The collection of layers, and methods for predicting and training.
/// T is the type of the inputs and outputs. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class Network {
public:
	Network(vector<Layer<T>> layers) : layers(layers) {}

	Tensor<T> forward(Tensor<T> inputs) {
		for (int i = 0; i < layers.size(); i++) {
			inputs = layers[i].forward(inputs);
		}
		
		return inputs;
	}

	/*Tensor<T> backward(Tensor<T> labels) {
		for (int i = 0; i < layers.size(); i++) {
			inputs = layers[i].forward(inputs);
		}

		return inputs;
	}*/

private:
	vector<Layer<T>> layers;
};
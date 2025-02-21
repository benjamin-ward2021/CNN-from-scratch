#pragma once

#include "Layer.cuh"
#include "FullyConnected.cuh"
#include "ReLU.cuh"
#include "SoftmaxCrossEntropy.cuh"
#include "Tensor.cuh"

using std::vector, std::unique_ptr;

/// <summary>
/// The collection of layers, and methods for predicting and training.
/// T is the type of the inputs and outputs. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class Network {
public:
	Network(vector<unique_ptr<Layer<T>>> layers) : layers(std::move(layers)) {}

	/// <summary>
	/// Performs forward propagation over all the layers.
	/// </summary>
	/// <param name="inputs"></param>
	/// <returns></returns>
	Tensor<T> forward(const Tensor<T> &inputs) {
		Tensor<T> outputs = inputs;
		for (int i = 0; i < layers.size(); i++) {
			outputs = layers[i]->forward(outputs);
		}
		
		return outputs;
	}

	/// <summary>
	/// Performs backward propagation over all the layers.
	/// </summary>
	/// <param name="labels"></param>
	void backward(const Tensor<T> &labels) {
		Tensor<T> gradients = labels;
		for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
			gradients = layers[i]->backward(gradients);
		}
	}

	/// <summary>
	/// Performs backward propagation over all the layers.
	/// </summary>
	/// <param name="labels"></param>
	void backward(const Tensor<int> &labels) {
		backward(labels.convert<T>());
	}

	/// <summary>
	/// Computes the loss.
	/// </summary>
	/// <param name="labels"></param>
	/// <returns></returns>
	T loss(const Tensor<T> &predicted, const Tensor<T> &labels) const {
		Tensor<T> predictedLog = predicted.log();
		T totalLoss = -labels.elementwiseMultiply(predictedLog).sum();
		T averageLoss = totalLoss / labels.getDims()[0];

		return averageLoss;
	}

	/// <summary>
	/// Computes the loss.
	/// </summary>
	/// <param name="labels"></param>
	/// <returns></returns>
	T loss(const Tensor<T> &predicted, const Tensor<int> &labels) const {
		return loss(predicted, labels.convert<T>());
	}

private:
	// Use a pointer to prevent object slicing
	vector<unique_ptr<Layer<T>>> layers;
};
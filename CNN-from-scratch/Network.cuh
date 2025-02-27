#pragma once

#include <concepts>

#include "Layer.cuh"
#include "FullyConnected.cuh"
#include "ReLU.cuh"
#include "SoftmaxCrossEntropy.cuh"
#include "Tensor.cuh"

/// <summary>
/// The collection of layers, and methods for predicting and training.
/// T is the type of the inputs and outputs. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T> requires std::floating_point<T>
class Network {
public:
	Network(std::vector<std::unique_ptr<Layer<T>>> layers) : layers(std::move(layers)) {}

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

	/// <summary>
	/// Computes the accuracy.
	/// </summary>
	/// <param name="labels"></param>
	/// <returns></returns>
	T accuracy(const Tensor<T> &predicted, const Tensor<T> &labels) const {
		assert(predicted.getDims() == labels.getDims());
		assert(predicted.getDims().size() == 2);

		const int batchSize = predicted.getDims()[0];
		const int numClasses = predicted.getDims()[1];
		T correct = 0;

		for (int i = 0; i < batchSize; ++i) {
			// Find predicted class
			int predClass = 0;
			T maxVal = predicted.get({ i, 0 });
			for (int j = 1; j < numClasses; ++j) {
				T val = predicted.get({ i, j });
				if (val > maxVal) {
					maxVal = val;
					predClass = j;
				}
			}

			// Find true class
			int trueClass = 0;
			T trueMax = labels.get({ i, 0 });
			for (int j = 1; j < numClasses; ++j) {
				T val = labels.get({ i, j });
				if (val > trueMax) {
					trueMax = val;
					trueClass = j;
				}
			}

			if (predClass == trueClass) correct += 1;
		}

		return correct / static_cast<T>(batchSize);
	}

	/// <summary>
	/// Computes the accuracy.
	/// </summary>
	/// <param name="labels"></param>
	/// <returns></returns>
	T accuracy(const Tensor<T> &predicted, const Tensor<int> &labels) const {
		return accuracy(predicted, labels.convert<T>());
	}

private:
	// Use a pointer to prevent object slicing
	std::vector<std::unique_ptr<Layer<T>>> layers;
};
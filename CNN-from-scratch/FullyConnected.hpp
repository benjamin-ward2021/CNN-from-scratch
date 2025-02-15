#pragma once

#include <random>

#include "Layer.hpp"
#include "Tensor.hpp"

using std::default_random_engine, std::normal_distribution, std::uniform_real_distribution;

/// <summary>
/// A fully connected layer. 
/// T is the type of the input, output, weights, and biases. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class FullyConnected : public Layer<T> {
public:
	FullyConnected(int inputSize, int outputSize, T learningRate, int rngSeed = 0, WeightInitializationHeuristic weightInitializationHeuristic = WeightInitializationHeuristic::heNormal)
		: weights(Tensor<T>({ inputSize,outputSize })), biases(Tensor<T>({ outputSize })), learningRate(learningRate), Layer<T>(Tensor<T>()) {
		default_random_engine randomEngine(rngSeed);

		assert(weightInitializationHeuristic == WeightInitializationHeuristic::heNormal ||
			weightInitializationHeuristic == WeightInitializationHeuristic::xavierUniform);

		if (weightInitializationHeuristic == WeightInitializationHeuristic::heNormal) {
			normal_distribution<T> normalDistribution(static_cast<T>(0), static_cast<T>(sqrt(2.0 / inputSize)));
			// Set weights according to normal he initialization
			weights.setToRandom(normalDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}

		else if (weightInitializationHeuristic == WeightInitializationHeuristic::xavierUniform) {
			T bound = static_cast<T>(sqrt(6.0 / (inputSize + outputSize)));
			uniform_real_distribution<T> uniformDistribution(-bound, bound);
			// Set weights according to uniform xavier initialization
			weights.setToRandom(uniformDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}
	}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="input">A tensor of inputs. Typically either dims = { batchSize,inputSize } in the case of layers like FullyConnected,
	/// or dims = { batchSize,height,width,depth } for layers like Convolutional or Pooling. Note that this function mutates input.</param>
	/// <returns>A 2d tensor of dims = { batchSize,outputSize }</returns>
	Tensor<T> forward(Tensor<T> &input) override {
		// Input dims will become = { batchSize,inputSize }
		input.reverseFlattenTo2d();
		this->input = input;

		// If Y = outputs, W = weights, X = inputs, B = biases...
		// Y = X * W + B
		// output dims will be = { batchSize,outputSize }
		Tensor<T> output = input.matrixMultiply<T>(weights);
		output.broadcastAddInPlace(biases);
		return output;
	}

	/// <summary>
	/// Performs backward propagation and updates weights and biases.
	/// </summary>
	/// <param name="gradWrtOutput">The gradient of loss with respect to the output of this layer. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the input of this layer. Dims = { batchSize,inputSize }.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutput) override {
		// Note that we have to explicitly use the "this" keyword with input since it is a dependent name.
		// See https://stackoverflow.com/questions/1527849/how-do-you-understand-dependent-names-in-c
		assert(this->input != Tensor<T>());

		// If L = loss, Y = outputs, W = weights, X = inputs, B = biases...
		// dL/dW = dL/dY * dY/dW. Since Y = X * W + B, dY/dW = X. So dL/dW = transpose(X) * dL/dY.
		// input has dims = { batchSize,inputSize }, so input.transpose() has dims = { inputSize,batchSize }
		// gradWrtWeights (gradient of loss with respect to weights) has dims = { intputSize,outputSize }
		Tensor<T> gradWrtWeights = this->input.transpose().matrixMultiply<T>(gradWrtOutput);

		// If L = loss, Y = outputs, W = weights, X = inputs, B = biases...
		// dL/dB = dL/dY * dY/dB. Since Y = X * W + B, dY/dB = 1. So dL/dB = 1 * dL/dY.
		// gradWrtBiases (gradient of loss with respect to biases) has dims = { outputSize }
		Tensor<T> gradWrtBiases = gradWrtOutput.matrixColumnSum();

		// If L = loss, Y = outputs, W = weights, X = inputs, B = biases...
		// dL/dX = dL/dY * dY/dX. Since Y = X * W + B, dY/dX = W. So dL/dX = dL/dY * transpose(W).
		// gradWrtInput (gradient of loss with respect to input) has dims = { batchSize,inputSize }
		Tensor<T> gradWrtInput = gradWrtOutput.matrixMultiply<T>(weights.transpose());

		// Update weights and biases
		weights.elementwiseSubtractInPlace(gradWrtWeights.scalarMultiply(learningRate));
		biases.elementwiseSubtractInPlace(gradWrtBiases.scalarMultiply(learningRate));

		return gradWrtInput;
	}

	// TODO: These functions
	void save() override {

	}
	void load() override {

	}

private:
	// 2d tensor with dims = { inputSize,outputSize }
	Tensor<T> weights;

	// 1d tensor with dims = { outputSize }
	Tensor<T> biases;

	T learningRate;
};
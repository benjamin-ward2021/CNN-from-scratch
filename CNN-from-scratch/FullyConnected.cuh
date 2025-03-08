#pragma once

#include <random>
#include <cassert>
#include <concepts>

#include "Layer.cuh"
#include "Tensor.cuh"

/// <summary>
/// A fully connected layer. 
/// T is the type of the input, output, weights, and biases. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T> requires std::floating_point<T>
class FullyConnected : public Layer<T> {
public:
	FullyConnected(int outputSize, T learningRate, int rngSeed, WeightInitializationHeuristic weightInitializationHeuristic = heNormal)
		: outputSize(outputSize), 
		originalInputsDims(std::vector<int>()),
		learningRate(learningRate),
		randomEngine(rngSeed), 
		weightInitializationHeuristic(weightInitializationHeuristic) {}

	void initialize(const std::vector<int> &inputDims) override {
		assert(weightInitializationHeuristic == heNormal || weightInitializationHeuristic == xavierUniform);

		inputSize = 1;
		for (int dim : inputDims) {
			inputSize *= dim;
		}

		weights = Tensor<T>({ inputSize,outputSize });
		biases = Tensor<T>({ outputSize });
		flattenedInputs = Tensor<T>();

		if (weightInitializationHeuristic == heNormal) {
			std::normal_distribution<T> normalDistribution(static_cast<T>(0), static_cast<T>(sqrt(2.0 / inputSize)));
			// Set weights according to normal he initialization
			weights.setToRandom(normalDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}

		else if (weightInitializationHeuristic == xavierUniform) {
			T bound = static_cast<T>(sqrt(6.0 / (inputSize + outputSize)));
			std::uniform_real_distribution<T> uniformDistribution(-bound, bound);
			// Set weights according to uniform xavier initialization
			weights.setToRandom(uniformDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}
	}

	std::vector<int> getOutputDims() const override {
		return { outputSize };
	}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="inputs">A tensor of inputs. Typically either dims = { batchSize,inputSize } or { batchSize,height,width,depth }. 
	/// Note that this function mutates inputs.</param>
	/// <returns>A 2d tensor of dims = { batchSize,outputSize }</returns>
	Tensor<T> forward(Tensor<T> &inputs) override {
		const int batchSize = inputs.getDims()[0];

		originalInputsDims = inputs.getDims();
		inputs.reverseFlattenTo2d();
		assert((inputs.getDims() == std::vector<int>{ batchSize,inputSize }));
		flattenedInputs = inputs;

		// If Y = outputs, W = weights, X = inputs, B = biases...
		// Y = X * W + B
		Tensor<T> outputs = flattenedInputs.matrixMultiplyGPU(weights);
		assert((outputs.getDims() == std::vector<int>{ batchSize,outputSize }));
		outputs.elementwiseAddInPlace(biases.broadcast(outputs.getDims()));
		return outputs;
	}

	/// <summary>
	/// Performs backward propagation and updates weights and biases.
	/// </summary>
	/// <param name="gradWrtOutputs">The gradient of loss with respect to the outputs of this layer. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the inputs of this layer. Dims = { batchSize,inputSize }.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutputs) override {
		const int batchSize = gradWrtOutputs.getDims()[0];

		assert(flattenedInputs != Tensor<T>());
		assert((flattenedInputs.getDims() == std::vector<int>{ batchSize,inputSize }));
		assert((gradWrtOutputs.getDims() == std::vector<int>{ batchSize,outputSize }));

		// If L = loss, Y = outputs, W = weights, X = inputs, B = biases...
		// dL/dW = dL/dY * dY/dW. Since Y = X * W + B, dY/dW = X. So dL/dW = transpose(X) * dL/dY.
		Tensor<T> gradWrtWeights = flattenedInputs.transpose().matrixMultiplyGPU(gradWrtOutputs);
		assert((gradWrtWeights.getDims() == std::vector<int>{ inputSize,outputSize }));

		// dL/dB = dL/dY * dY/dB. Since Y = X * W + B, dY/dB = 1. So dL/dB = 1 * dL/dY.
		Tensor<T> gradWrtBiases = gradWrtOutputs.sum(0);
		assert((gradWrtBiases.getDims() == std::vector<int>{ outputSize }));

		// dL/dX = dL/dY * dY/dX. Since Y = X * W + B, dY/dX = W. So dL/dX = dL/dY * transpose(W).
		Tensor<T> gradWrtFlattenedInputs = gradWrtOutputs.matrixMultiplyGPU(weights.transpose());
		assert((gradWrtFlattenedInputs.getDims() == std::vector<int>{ batchSize,inputSize }));

		// Update weights and biases
		weights.elementwiseSubtractInPlace(gradWrtWeights.scalarMultiply(learningRate));
		biases.elementwiseSubtractInPlace(gradWrtBiases.scalarMultiply(learningRate));

		// Return the gradient of loss with respect to inputs in the original dimensions so that the previous layer can use it
		Tensor<T> gradWrtInputs = gradWrtFlattenedInputs;
		gradWrtInputs.reinterpretDims(originalInputsDims);
		return gradWrtInputs;
	}

	// TODO: These functions
	void save() override {

	}
	void load() override {

	}

private:
	// Amount of incoming nodes
	int inputSize;

	// Amount of outgoing nodes
	int outputSize;
	
	// 2d tensor with dims = { batchSize,inputSize }
	Tensor<T> flattenedInputs;

	std::vector<int> originalInputsDims;

	// 2d tensor with dims = { inputSize,outputSize }
	Tensor<T> weights;

	// 1d tensor with dims = { outputSize }
	Tensor<T> biases;

	// Multiplied with the gradients when updating weights and biases. Should be between 0 and 1
	T learningRate;

	// Used for weight initialization
	std::default_random_engine randomEngine;

	// Determines the weight initialization scheme
	WeightInitializationHeuristic weightInitializationHeuristic;
};
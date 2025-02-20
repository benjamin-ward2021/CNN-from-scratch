#pragma once

#include <random>
#include <cassert>

#include "Layer.cuh"
#include "Tensor.cuh"

using std::default_random_engine, std::normal_distribution, std::uniform_real_distribution;

/// <summary>
/// A fully connected layer. 
/// T is the type of the input, output, weights, and biases. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class FullyConnected : public Layer<T> {
public:
	FullyConnected(int inputSize, int outputSize, T learningRate, bool useGPU = true, int rngSeed = 0, WeightInitializationHeuristic weightInitializationHeuristic = heNormal)
		: inputSize(inputSize), 
		outputSize(outputSize), 
		weights(Tensor<T>({ inputSize,outputSize })), 
		biases(Tensor<T>({ outputSize })), 
		inputs(Tensor<T>()), 
		learningRate(learningRate), 
		useGPU(useGPU) {

		assert(weightInitializationHeuristic == heNormal || weightInitializationHeuristic == xavierUniform);
		default_random_engine randomEngine(rngSeed);

		if (weightInitializationHeuristic == heNormal) {
			normal_distribution<T> normalDistribution(static_cast<T>(0), static_cast<T>(sqrt(2.0 / inputSize)));
			// Set weights according to normal he initialization
			weights.setToRandom(normalDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}

		else if (weightInitializationHeuristic == xavierUniform) {
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
	/// <param name="inputs">A tensor of inputs. Typically either dims = { batchSize,inputSize } or { batchSize,height,width,depth }. 
	/// Note that this function mutates inputs.</param>
	/// <returns>A 2d tensor of dims = { batchSize,outputSize }</returns>
	Tensor<T> forward(Tensor<T> &inputs) override {
		const int batchSize = inputs.getDims()[0];

		inputs.reverseFlattenTo2d();
		assert((inputs.getDims() == vector<int>{ batchSize,inputSize }));
		this->inputs = inputs;

		// If Y = outputs, W = weights, X = inputs, B = biases...
		// Y = X * W + B
		Tensor<T> outputs = useGPU ? inputs.matrixMultiplyGPU(weights) : inputs.matrixMultiply(weights);
		assert((outputs.getDims() == vector<int>{ batchSize,outputSize }));
		outputs.broadcastAddInPlace(biases);
		return outputs;
	}

	/// <summary>
	/// Performs backward propagation and updates weights and biases.
	/// </summary>
	/// <param name="gradWrtOutputs">The gradient of loss with respect to the outputs of this layer. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the inputs of this layer. Dims = { batchSize,inputSize }.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutputs) override {
		const int batchSize = gradWrtOutputs.getDims()[0];

		assert(inputs != Tensor<T>());
		assert((inputs.getDims() == vector<int>{ batchSize,inputSize }));
		assert((gradWrtOutputs.getDims() == vector<int>{ batchSize,outputSize }));

		// If L = loss, Y = outputs, W = weights, X = inputs, B = biases...
		// dL/dW = dL/dY * dY/dW. Since Y = X * W + B, dY/dW = X. So dL/dW = transpose(X) * dL/dY.
		Tensor<T> gradWrtWeights = useGPU ? inputs.transpose().matrixMultiplyGPU(gradWrtOutputs) : inputs.transpose().matrixMultiply(gradWrtOutputs);
		assert((gradWrtWeights.getDims() == vector<int>{ inputSize,outputSize }));

		// dL/dB = dL/dY * dY/dB. Since Y = X * W + B, dY/dB = 1. So dL/dB = 1 * dL/dY.
		Tensor<T> gradWrtBiases = gradWrtOutputs.sum(0);
		assert((gradWrtBiases.getDims() == vector<int>{ outputSize }));

		// dL/dX = dL/dY * dY/dX. Since Y = X * W + B, dY/dX = W. So dL/dX = dL/dY * transpose(W).
		Tensor<T> gradWrtInputs = useGPU ? gradWrtOutputs.matrixMultiplyGPU(weights.transpose()) : gradWrtOutputs.matrixMultiply(weights.transpose());
		assert((gradWrtWeights.getDims() == vector<int>{ batchSize,inputSize }));

		// Update weights and biases
		weights.elementwiseSubtractInPlace(gradWrtWeights.scalarMultiply(learningRate));
		biases.elementwiseSubtractInPlace(gradWrtBiases.scalarMultiply(learningRate));

		// Return the gradient of loss with respect to inputs so that the previous layer can use it
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

	// TODO: Should this be the flattened version? Probably all layers should be consistent in if they modify inputs before saving it.
	Tensor<T> inputs;

	// 2d tensor with dims = { inputSize,outputSize }
	Tensor<T> weights;

	// 1d tensor with dims = { outputSize }
	Tensor<T> biases;

	// Multiplied with the gradients when updating weights and biases. Should be between 0 and 1
	T learningRate;

	// Determines whether the GPU is used for calculations like matrix multiplications
	bool useGPU;
};
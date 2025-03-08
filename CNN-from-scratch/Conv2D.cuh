#pragma once

#include <random>
#include <cassert>
#include <concepts>

#include "Layer.cuh"
#include "Tensor.cuh"
#include "Kernels.cuh"

/// <summary>
/// A 2D convolutional layer. 
/// T is the type of the input, output, weights, and biases. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T> requires std::floating_point<T>
class Conv2D : public Layer<T> {
public:
	Conv2D(int outputChannels, int kernelSize, T learningRate, int rngSeed, 
		int stride = 1, int padding = 0, WeightInitializationHeuristic weightInitializationHeuristic = heNormal) 
		: outputChannels(outputChannels), 
		kernelSize(kernelSize),
		learningRate(learningRate), 
		randomEngine(rngSeed),
		stride(stride), 
		padding(padding),
		weightInitializationHeuristic(weightInitializationHeuristic) {}

	void initialize(const std::vector<int> &inputDims) override {
		assert(weightInitializationHeuristic == heNormal || weightInitializationHeuristic == xavierUniform);

		inputChannels = inputDims[0];
		outputHeight = (inputDims[1] + 2 * padding - kernelSize) / stride + 1;
		outputWidth = (inputDims[2] + 2 * padding - kernelSize) / stride + 1;

		weights = Tensor<T>({ outputChannels,inputChannels,kernelSize,kernelSize });
		biases = Tensor<T>({ outputChannels });
		inputs = Tensor<T>();

		if (weightInitializationHeuristic == heNormal) {
			std::normal_distribution<T> normalDistribution(static_cast<T>(0), static_cast<T>(sqrt(2.0 / (inputChannels * kernelSize * kernelSize))));
			// Set weights according to normal he initialization
			weights.setToRandom(normalDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}

		else if (weightInitializationHeuristic == xavierUniform) {
			T bound = static_cast<T>(sqrt(6.0 / (inputChannels + outputChannels)));
			std::uniform_real_distribution<T> uniformDistribution(-bound, bound);
			// Set weights according to uniform xavier initialization
			weights.setToRandom(uniformDistribution, randomEngine);

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			biases.setToZero();
		}
	}

	std::vector<int> getOutputDims() const override {
		return { outputChannels, outputHeight, outputWidth };
	}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="inputs">A tensor of inputs where dims = { batchSize,inputSize }.</param>
	/// <returns>A 4d tensor of dims = { batchSize,inputChannels,inputHeight,inputWidth }</returns>
	Tensor<T> forward(Tensor<T> &inputs) override {
		assert(inputs.getDims().size() == 4);
		this->inputs = inputs;
		Tensor<T> outputs = Tensor<T>::conv2dForwardGPU(inputs, weights, biases, stride, padding);
		return outputs;
	}

	/// <summary>
	/// Performs backward propagation and updates weights and biases.
	/// </summary>
	/// <param name="gradWrtOutputs">The gradient of loss with respect to the outputs of this layer. Dims = { batchSize,outputChannels,kernelSize,kernelSize }.</param>
	/// <returns>The gradient of loss with respect to the inputs of this layer. Dims = { batchSize,inputChannels,inputHeight,inputWidth }.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutputs) override {
		// Compute gradients
		Tensor<T> gradWrtInputs = Tensor<T>::conv2dGradInputsGPU(gradWrtOutputs, weights, inputs, stride, padding);
		Tensor<T> gradWrtWeights = Tensor<T>::conv2dGradWeightsGPU(inputs, gradWrtOutputs, weights, stride, padding);

		// Sum over batch, then height, then width (leaving only output channels)
		Tensor<T> gradWrtBiases = gradWrtOutputs.sum(0).sum(1).sum(1);

		weights = weights.elementwiseSubtract(gradWrtWeights.scalarMultiply(learningRate));
		biases = biases.elementwiseSubtract(gradWrtBiases.scalarMultiply(learningRate));

		return gradWrtInputs;
	}

	// TODO: These functions
	void save() override {

	}
	void load() override {

	}

private:
	// 4d tensor with dims = { batchSize,inputChannels,inputHeight,inputWidth }
	Tensor<T> inputs;

	// 4d tensor with dims = { outputChannels,inputChannels,kernelSize,kernelSize }
	Tensor<T> weights;

	// 1d tensor with dims = { outputChannels }
	Tensor<T> biases;

	// Number of incoming channels (Ex. if the previous layer was the input of an RBG image, this would be 3)
	int inputChannels;

	// Number of outgoing channels (Ex. number of filters)
	int outputChannels;

	int outputHeight;

	int outputWidth;

	// Size of the kernel. Assumes the kernel is a square
	int kernelSize;

	// The distance the filters move across the image on each step. The default would be 1
	int stride;

	// The amount of padded pixels along the edges of the input
	int padding;

	// Multiplied with the gradients when updating weights and biases. Should be between 0 and 1
	T learningRate;

	// Used for weight initialization
	std::default_random_engine randomEngine;

	// Determines the weight initialization scheme
	WeightInitializationHeuristic weightInitializationHeuristic;
};
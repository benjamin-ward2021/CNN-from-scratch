#pragma once

#include <concepts>

#include "Tensor.cuh"

/// <summary>
/// Interface for all layer types. 
/// (Ex. FullyConnected, Convolutional, ReLU).
/// All layers in an architecture must share the same type T.
/// </summary>
/// <typeparam name="T"></typeparam>
template<typename T> requires std::floating_point<T>
class Layer {
public:
	// Note inputDims does NOT include the batch dimension
	virtual void initialize(const std::vector<int> &inputDims) = 0;
	// Note that layers are allowed to mutate the inputs during forward propagation.
	virtual Tensor<T> forward(Tensor<T> &inputs) = 0;
	// gradWrtOutputs is the gradient of loss with respect to the outputs.
	// Returns the gradient of loss with respect to the inputs.
	virtual Tensor<T> backward(const Tensor<T> &gradWrtOutputs) = 0;
	// Used for setting the input dims of the next layer
	virtual std::vector<int> getOutputDims() const = 0;
	virtual void save() = 0;
	virtual void load() = 0;
};

// he recommended for ReLU. xavier recommended for sigmoid, tanh, softmax.
enum WeightInitializationHeuristic { heNormal, xavierUniform };
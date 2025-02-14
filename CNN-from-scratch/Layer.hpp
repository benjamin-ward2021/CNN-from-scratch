#pragma once

#include "Tensor.hpp"

/// <summary>
/// Base class for all layer types. 
/// (Ex. FullyConnected, Convolutional, ReLU).
/// All layers in an architecture must share the same type.
/// </summary>
/// <typeparam name="T"></typeparam>
template<typename T>
class Layer {
public:
	// Default constructor just initializes the input tensor.
	Layer(Tensor<T> input) : input(input) {}
	// Note that layers are allowed to mutate the input during forward propagation.
	virtual Tensor<T> forward(Tensor<T> &input) = 0;
	// gradWrtOutput is the gradient of loss with respect to the output.
	virtual Tensor<T> backward(const Tensor<T> &gradWrtOutput) = 0;
	virtual void save() = 0;
	virtual void load() = 0;

protected:
	Tensor<T> input;
};

// he recommended for ReLU. xavier recommended for sigmoid, tanh, softmax.
enum WeightInitializationHeuristic { heNormal, xavierUniform };
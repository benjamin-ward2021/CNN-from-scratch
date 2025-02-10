#pragma once

#include "Tensor.hpp"

/// <summary>
/// Base class for all layer types. 
/// (Ex. FullyConnected, Convolutional, ReLU).
/// All calculations are done using doubles, so no need to make this a class template.
/// </summary>
class Layer {
public:
	// he recommended for ReLU. xavier recommended for sigmoid, tanh, softmax.
	enum WeightInitializationHeuristic { heNormal, xavierUniform };
	virtual Tensor<double> forward(const Tensor<double> &input) = 0;
	virtual Tensor<double> backward() = 0;
	virtual void save() = 0;
	virtual void load() = 0;
};
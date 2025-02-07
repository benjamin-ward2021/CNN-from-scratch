#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
/// <summary>
/// A fully connected layer. 
/// T is the type of the weights and biases. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class FullyConnected : public Layer {
public:
	Tensor<double> forward(const Tensor<double> &input) override;
	Tensor<double> backward() override;
	void save() override;
	void load() override;
private:
	Tensor<T> weights;
	Tensor<T> bias;
};
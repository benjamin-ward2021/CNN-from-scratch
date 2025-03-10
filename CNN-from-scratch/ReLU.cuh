#pragma once

#include <concepts>

#include "Layer.cuh"
#include "Tensor.cuh"

/// <summary>
/// A ReLU (Rectified Linear Unit) layer. 
/// T is the type of the inputs and outputs. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T> requires std::floating_point<T>
class ReLU : public Layer<T> {
public:
	ReLU() = default;

	void initialize(const std::vector<int> &inputDims) override {
		inputs = Tensor<T>();
		this->inputDims = inputDims;
	}

	std::vector<int> getOutputDims() const override {
		return inputDims;
	}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="inputs">A tensor of inputs. Typically either dims = { batchSize,inputSize } or { batchSize,height,width,depth }.</param>
	/// <returns>A tensor with dims = inputs.dims</returns>
	Tensor<T> forward(Tensor<T> &inputs) override {
		this->inputs = inputs;
		return inputs.relu();
	}

	/// <summary>
	/// Performs backward propagation.
	/// </summary>
	/// <param name="gradWrtOutputs">The gradient of loss with respect to the outputs of this layer. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the inputs of this layer. Dims = inputs.dims.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutputs) override {
		assert(inputs != Tensor<T>());
		Tensor<T> gradWrtInputs = gradWrtOutputs.elementwiseMultiply(inputs.reluDerivative());
		assert((inputs.getDims() == gradWrtInputs.getDims()));
		return gradWrtInputs;
	}

	// TODO: These functions (ReLU layers don't have anything to save or load...)
	void save() override {

	}
	void load() override {

	}

private:
	Tensor<T> inputs;
	std::vector<int> inputDims;
};
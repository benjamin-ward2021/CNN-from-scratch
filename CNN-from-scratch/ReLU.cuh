#pragma once

#include "Layer.cuh"
#include "Tensor.cuh"

/// <summary>
/// A ReLU (Rectified Linear Unit) layer. 
/// T is the type of the input and output. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class ReLU : public Layer<T> {
public:
	ReLU() : Layer<T>(Tensor<T>()) {}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="input">A tensor of inputs. Typically either dims = { batchSize,inputSize } in the case of layers like FullyConnected,
	/// or dims = { batchSize,height,width,depth } for layers like Convolutional or Pooling.</param>
	/// <returns>A tensor with dims = input.dims</returns>
	Tensor<T> forward(Tensor<T> &input) override {
		this->input = input;
		return input.relu();
	}

	/// <summary>
	/// Performs backward propagation.
	/// </summary>
	/// <param name="gradWrtOutput">The gradient of loss with respect to the output of this layer. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the input of this layer. Dims = input.dims.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutput) override {
		// Note that we have to explicitly use the "this" keyword with input since it is a dependent name.
		// See https://stackoverflow.com/questions/1527849/how-do-you-understand-dependent-names-in-c
		assert(this->input != Tensor<T>());
		// gradWrtInput(gradient of loss with respect to input) has dims = input.dims
		Tensor<T> gradWrtInput = gradWrtOutput.elementwiseMultiply(this->input.reluDerivative());
		return gradWrtInput;
	}

	// TODO: These functions (ReLU layers don't have anything to save or load...)
	void save() override {

	}
	void load() override {

	}

private:
	// 2d tensor with dims = { inputSize,outputSize }
	Tensor<T> weights;

	// 1d tensor with dims = { outputSize }
	Tensor<T> biases;
};
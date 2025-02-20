#pragma once

#include "Layer.cuh"
#include "Tensor.cuh"

/// <summary>
/// A softmax layer. Converts inputs into a corresponding probability distribution.
/// T is the type of the input and output. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class Softmax : public Layer<T> {
public:
	Softmax() : inputs(Tensor<T>()) {}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="inputs">A tensor of inputs. Typically either dims = { batchSize,inputSize } in the case of layers like FullyConnected,
	/// or dims = { batchSize,height,width,depth } for layers like Convolutional or Pooling.</param>
	/// <returns>A tensor with dims = input.dims</returns>
	Tensor<T> forward(Tensor<T> &inputs) override {
		this->inputs = inputs;
		return inputs.softmax();
	}

	/// <summary>
	/// Performs backward propagation.
	/// </summary>
	/// <param name="labels">The labels that we are trying to predict. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the inputs of this layer. Dims = inputs.dims.</returns>
	Tensor<T> backward(const Tensor<T> &labels) override {
		assert(inputs != Tensor<T>());
		// The gradient of softmax is simply predicted - labels.
		// See 
		//Tensor<T> gradWrtInputs = inputs.elementwiseSubtract(labels);
		//return gradWrtInputs;
	}

	// TODO: These functions (ReLU layers don't have anything to save or load...)
	void save() override {

	}
	void load() override {

	}

private:
	Tensor<T> inputs;
};
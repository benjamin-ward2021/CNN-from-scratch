#pragma once

#include "Layer.cuh"
#include "Tensor.cuh"

/// <summary>
/// A combined softmax and cross entropy loss layer.
/// T is the type of the input and output. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class SoftmaxCrossEntropy : public Layer<T> {
public:
	SoftmaxCrossEntropy() : outputs(Tensor<T>()) {}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="inputs">A tensor of inputs. Typically either dims = { batchSize,inputSize } in the case of layers like FullyConnected,
	/// or dims = { batchSize,height,width,depth } for layers like Convolutional or Pooling.</param>
	/// <returns>A tensor with dims = input.dims</returns>
	Tensor<T> forward(Tensor<T> &inputs) override {
		outputs = inputs.softmax(1);
		return outputs;
	}

	/// <summary>
	/// Performs backward propagation, using cross entropy loss.
	/// </summary>
	/// <param name="labels">The labels that we are trying to predict. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the inputs of this layer. Dims = inputs.dims.</returns>
	Tensor<T> backward(const Tensor<T> &labels) override {
		assert(outputs != Tensor<T>());
		// The gradient of softmax is simply predicted - labels when you combine it with cross entropy loss.
		// We divide it by the batchSize; otherwise a higher batch size would mean faster learning.
		int batchSize = labels.getDims()[0];
		Tensor<T> gradWrtInputs = outputs.elementwiseSubtract(labels).scalarDivide(static_cast<T>(batchSize));
		return gradWrtInputs;
	}

	/// <summary>
	/// Computes the loss.
	/// </summary>
	/// <param name="labels"></param>
	/// <returns></returns>
	T loss(const Tensor<T> &labels) {
		Tensor<T> softmaxLog = outputs.log();
		T totalLoss = -labels.elementwiseMultiply(softmaxLog).sum();
		T averageLoss = totalLoss / labels.getDims()[0];

		return averageLoss;
	}

	// TODO: These functions (softmax layers don't have anything to save or load...)
	void save() override {

	}
	void load() override {

	}

private:
	Tensor<T> outputs;
};
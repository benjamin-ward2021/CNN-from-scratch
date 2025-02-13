#pragma once

#include <random>

#include "Layer.hpp"
#include "Tensor.hpp"

using std::default_random_engine, std::normal_distribution, std::uniform_real_distribution;

/// <summary>
/// A fully connected layer. 
/// T is the type of the weights and biases. (Ex. float, double).
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class FullyConnected : public Layer<T> {
public:
	FullyConnected(int inputSize, int outputSize, int rngSeed = 0, WeightInitializationHeuristic weightInitializationHeuristic = WeightInitializationHeuristic::heNormal)
		: weights(Tensor<T>({ inputSize,outputSize })), biases(Tensor<T>({ outputSize })), input(Tensor<T>({ 0 })) {
		default_random_engine randomEngine(rngSeed);

		assert(weightInitializationHeuristic == WeightInitializationHeuristic::heNormal ||
			weightInitializationHeuristic == WeightInitializationHeuristic::xavierUniform);

		if (weightInitializationHeuristic == WeightInitializationHeuristic::heNormal) {
			normal_distribution<T> normalDistribution(static_cast<T>(0), static_cast<T>(sqrt(2.0 / inputSize)));
			// Set weights according to normal he initialization
			for (int i = 0; i < inputSize; i++) {
				for (int j = 0; j < outputSize; j++) {
					weights.set({ i,j }, normalDistribution(randomEngine));
				}
			}

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			for (int i = 0; i < outputSize; i++) {
				biases.set({ i }, static_cast<T>(0));
			}
		}

		else if (weightInitializationHeuristic == WeightInitializationHeuristic::xavierUniform) {
			T bound = static_cast<T>(sqrt(6.0 / (inputSize + outputSize)));
			uniform_real_distribution<T> uniformDistribution(-bound, bound);
			// Set weights according to uniform xavier initialization
			for (int i = 0; i < inputSize; i++) {
				for (int j = 0; j < outputSize; j++) {
					weights.set({ i,j }, uniformDistribution(randomEngine));
				}
			}

			// Set biases to 0. Note that this is redundant since when the tensor is created, all values are initialized to 0
			for (int i = 0; i < outputSize; i++) {
				biases.set({ i }, static_cast<T>(0));
			}
		}
	}

	/// <summary>
	/// Performs forward propagation.
	/// </summary>
	/// <param name="input">A tensor of inputs. Typically either dims = { batchSize,inputSize } in the case of layers like FullyConnected,
	/// or dims = { batchSize,height,width,depth } for layers like Convolutional or Pooling. Note that this function mutates input.</param>
	/// <returns>A 2d tensor of dims = { batchSize,outputSize }</returns>
	Tensor<T> forward(Tensor<T> &input) override {
		// Input dims will become = { batchSize,inputSize }
		input.reverseFlattenTo2d();
		this->input = input;
		// Output dims will be = { batchSize,outputSize }
		Tensor<T> output = input.matrixMultiply<T>(weights);
		output.broadcastAddInPlace(biases);
		return output;
	}

	/// <summary>
	/// Performs backward propagation and updates weights and biases.
	/// </summary>
	/// <param name="gradWrtOutput">The gradient of loss with respect to the output of this layer. Dims = { batchSize,outputSize }.</param>
	/// <returns>The gradient of loss with respect to the input of this layer. Dims = { batchSize,inputSize }.</returns>
	Tensor<T> backward(const Tensor<T> &gradWrtOutput) override {
		assert(input != Tensor<T>({ 0 }));
		// input has dims = { batchSize,inputSize }, so input.transpose() has dims = { inputSize,batchSize }
		// gradWrtWeights (gradient of loss with respect to weights) has dims = { intputSize,outputSize }
		Tensor<T> gradWrtWeights = input.transpose().matrixMultiply<T>(gradWrtOutput);

		// gradWrtBiases (gradient of loss with respect to biases) has dims = { outputSize }
		Tensor<T> gradWrtBiases = gradWrtOutput.matrixColumnSum();

		// gradWrtInput (gradient of loss with respect to input) has dims = { batchSize,inputSize }
		// Make sure to initialize this before updating weights
		Tensor<T> gradWrtInput = gradWrtOutput.matrixMultiply<T>(weights.transpose());

		// TODO: Update weights

		return gradWrtInput;
	}

	// TODO: These functions
	void save() override {

	}
	void load() override {

	}

private:
	// 2d tensor with dims = { inputSize,outputSize }
	Tensor<T> weights;

	// 1d tensor with dims = { outputSize }
	Tensor<T> biases;

	// 2d tensor with dims = { batchSize,inputSize }
	Tensor<T> input;
};
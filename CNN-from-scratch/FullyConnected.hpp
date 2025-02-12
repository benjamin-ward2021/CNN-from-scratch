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
		: weights(Tensor<T>({ inputSize,outputSize })), biases(Tensor<T>({ outputSize })) {
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
		// Output dims will be = { batchSize,outputSize }
		Tensor<T> output = input.matrixMultiply<T>(weights);
		output.broadcastAddInPlace(biases);
		return output;
	}

	// TODO: These functions
	Tensor<T> backward() override {
		return Tensor<T>({ 1 });
	}
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
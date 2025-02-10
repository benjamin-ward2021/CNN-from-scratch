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
class FullyConnected : public Layer {
public:
	FullyConnected(int inputSize, int outputSize, int rngSeed, Layer::WeightInitializationHeuristic weightInitializationHeuristic) : 
		weights(Tensor<T>({ inputSize,outputSize })), biases(Tensor<T>({ outputSize })) {
		default_random_engine randomEngine(rngSeed);

		assert(weightInitializationHeuristic == Layer::WeightInitializationHeuristic::heNormal ||
			weightInitializationHeuristic == Layer::WeightInitializationHeuristic::xavierUniform);

		if (weightInitializationHeuristic == Layer::WeightInitializationHeuristic::heNormal) {
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

		else if (weightInitializationHeuristic == Layer::WeightInitializationHeuristic::xavierUniform) {
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

	// TODO: These functions
	Tensor<double> forward(const Tensor<double> &input) override {
		return Tensor<double>({ 1 });
	}
	Tensor<double> backward() override {
		return Tensor<double>({ 1 });
	}
	void save() override {

	}
	void load() override {

	}
private:
	Tensor<T> weights;
	Tensor<T> biases;
};
#pragma once

#include <random>
#include <concepts>

#include "Tensor.cuh"

/// <summary>
/// Generates data and labels for the XOR problem.
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T> requires std::floating_point<T>
class XORGenerator {
public:
	XORGenerator(int seed) : randomEngine(seed), uniformDistribution(static_cast<T>(0), static_cast<T>(1)) {}

	/// <summary>
	/// Generates two numbers between 0 and 1, and gives 
	/// a label of { 1,0 } if they are on different sides of 0.5, and { 0,1 } if they aren't.
	/// </summary>
	/// <param name="totalNumSamples"></param>
	void generate(int totalNumSamples) {
		inputs = Tensor<T>({ totalNumSamples,2 });
		labels = Tensor<int>({ totalNumSamples,2 });
		for (int i = 0; i < totalNumSamples; i++) {
			T x1 = uniformDistribution(randomEngine);
			T x2 = uniformDistribution(randomEngine);
			inputs.set({ i,0 }, x1);
			inputs.set({ i,1 }, x2);
			bool isXor = x1 >= static_cast<T>(0.5) != x2 >= static_cast<T>(0.5);
			if (isXor) {
				labels.set({ i,0 }, 1);
				labels.set({ i,1 }, 0);
			}

			else {
				labels.set({ i,0 }, 0);
				labels.set({ i,1 }, 1);
			}
		}
	}

	[[nodiscard]]
	Tensor<T> getInputs() const {
		return inputs;
	}

	[[nodiscard]]
	Tensor<int> getLabels() const {
		return labels;
	}
private:
	std::default_random_engine randomEngine;
	std::uniform_real_distribution<T> uniformDistribution;

	// Dims = { totalNumSamples,2 } since there are two inputs for XOR problem
	Tensor<T> inputs;
	// Dims = { totalNumSamples,2 } since we are one-hot encoding the labels.
	Tensor<int> labels;
};
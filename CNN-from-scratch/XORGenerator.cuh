#pragma once

#include <random>

#include "Tensor.cuh"

using std::default_random_engine, std::uniform_real_distribution;

template <typename T>
class XORGenerator {
public:
	XORGenerator(int seed) : randomEngine(seed), uniformDistribution(static_cast<T>(-1), static_cast<T>(1)) {}
	void generate(int totalNumSamples) {
		inputs = Tensor<T>({ totalNumSamples,2 });
		labels = Tensor<int>({ totalNumSamples,2 });
		for (int i = 0; i < totalNumSamples; i++) {
			T x1 = uniformDistribution(randomEngine);
			T x2 = uniformDistribution(randomEngine);
			inputs.set({ i,0 }, x1);
			inputs.set({ i,1 }, x2);
			bool isXor = (x1 >= 0 && x2 >= 0) || (x1 < 0 && x2 < 0);
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
	default_random_engine randomEngine;
	uniform_real_distribution<T> uniformDistribution;

	// Dims = { totalNumSamples,2 } since there are two inputs for XOR problem
	Tensor<T> inputs;
	// Dims = { totalNumSamples,2 } since we are one-hot encoding the labels.
	// if { i,0 } = 1 then the inputs share the same sign, if { i,1 } = 1 then they don't
	Tensor<int> labels;
};
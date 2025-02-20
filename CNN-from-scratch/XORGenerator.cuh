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
		labels = Tensor<T>({ totalNumSamples,1 });
		for (int i = 0; i < totalNumSamples; i++) {
			T x1 = uniformDistribution(randomEngine);
			T x2 = uniformDistribution(randomEngine);
			inputs.set({ i,0 }, x1);
			inputs.set({ i,1 }, x2);
			bool isXor = (x1 >= 0 && x2 >= 0) || (x1 < 0 && x2 < 0);
			labels.set({ i }, isXor ? static_cast<T>(0) : static_cast<T>(1));
		}
	}

	[[nodiscard]]
	Tensor<T> getInputs() const {
		return inputs;
	}

	[[nodiscard]]
	Tensor<T> getLabels() const {
		return labels;
	}
private:
	default_random_engine randomEngine;
	uniform_real_distribution<T> uniformDistribution;

	Tensor<T> inputs;
	Tensor<T> labels;
};
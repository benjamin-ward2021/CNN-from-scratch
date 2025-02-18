#pragma once

#include <random>

#include "Tensor.cuh"

using std::default_random_engine, std::uniform_real_distribution;

template <typename T>
class XORGenerator {
public:
	XORGenerator(int seed) : randomEngine(seed), uniformDistribution(static_cast<T>(-1), static_cast<T>(1)) {}
	void generate(int totalNumSamples) {
		input = Tensor<T>({ totalNumSamples,2 });
		output = Tensor<T>({ totalNumSamples,1 });
		for (int i = 0; i < totalNumSamples; i++) {
			T x1 = uniformDistribution(randomEngine);
			T x2 = uniformDistribution(randomEngine);
			input.set({ i,0 }, x1);
			input.set({ i,1 }, x2);
			bool isXor = (x1 >= 0 && x2 >= 0) || (x1 < 0 && x2 < 0);
			output.set({ i }, isXor ? static_cast<T>(0) : static_cast<T>(1));
		}
	}

	[[nodiscard]]
	Tensor<T> getInput() const {
		return input;
	}

	[[nodiscard]]
	Tensor<T> getOutput() const {
		return output;
	}
private:
	default_random_engine randomEngine;
	uniform_real_distribution<T> uniformDistribution;

	Tensor<T> input;
	Tensor<T> output;
};
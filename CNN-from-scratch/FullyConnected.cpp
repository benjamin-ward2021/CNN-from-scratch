#include "FullyConnected.hpp"

#include "Tensor.hpp"

template <typename T>
Tensor<double> FullyConnected<T>::forward(const Tensor<double> &input) {
	return input;
}

template <typename T>
Tensor<double> FullyConnected<T>::backward() {

}
template <typename T>
void FullyConnected<T>::save() {

}

template <typename T>
void FullyConnected<T>::load() {

}
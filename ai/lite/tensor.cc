#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>

using namespace std;

template <typename Dtype>
class Tensor {
public:
	std::vector<size_t> shape;
	Dtype* data;

	size_t getDim() const {
		return shape.size();
	}

	size_t getSize() const {
		if (shape.empty()) return 0;
		size_t size = 1;
		for (auto i: shape) size *= i;
		return size;
	}

	size_t getSize(int i) const {
		return (i >= shape.size()) ? -1 : shape[i];
	}

	size_t initMem() {
		data = (Dtype*)malloc(sizeof(Dtype)*getSize());
		memset(data, 0x0, sizeof(Dtype)*getSize());
	}

	void zero_() { // inplace
		memset(data, 0x0, sizeof(Dtype)*getSize());
	}
};

template <typename Dtype>
class Vector : public Tensor<Dtype> {
public:
	Vector(size_t length) {
		this->shape.push_back(length);
		this->initMem();
	}

	Dtype* at(size_t offset) {
		return this->data + offset;
	}

	void dump() {
		std::cout << "[";
		for (int i = 0; i < this->getSize(0); i++)
			printf(" %.3f", *this->at(i));
		std::cout << " ]" << std::endl;
	}
};

template <typename Dtype>
class Matrix : public Tensor<Dtype> {
public:
	Matrix(size_t row, size_t col) {
		this->shape.push_back(row);
		this->shape.push_back(col);
		this->initMem();
	}

	Dtype* at(size_t row, size_t col) {
		return this->data + row*this->shape[1] + col;
	}

	void dump() {
		std::cout << "[" << std::endl;;
		for (int i = 0; i < this->getSize(0); i++) {
			std::cout << "  [";
			for (int j = 0; j < this->getSize(1); j++) {
				printf(" %.3f", *this->at(i, j));
			}
			std::cout << " ]" << std::endl;

		}
		std::cout << "]" << std::endl;
	}
};

template <typename Dtype>
void
GEMM(Dtype alpha, Matrix<Dtype>* A, Dtype beta, Matrix<Dtype>* B,
		Dtype gamma, Matrix<Dtype>* C)
{
	// check shape
	assert(A->shape[1] == B->shape[0]);
	assert(A->shape[0] == C->shape[0] && B->shape[1] == C->shape[1]);
	// do gemm
	for (int i = 0; i < C->shape[0]; i++) {
		for (int j = 0; j < C->shape[1]; j++) {
			for (int k = 0; k < A->shape[1]; k++) {
				*C->at(i, j) = gamma * *C->at(i, j) +
					alpha * *A->at(i, k) + beta * *B->at(k, j);
			}
		}
	}
}

#if defined(LITE_TEST_TENSOR)
#include "dataloader.cc"
int
main(void)
{
	Matrix<double> data (8, 17);
	Vector<double> label (10);

	lite_hdf5_read("demo.h5", "data", 0, 0, 8, 17, data.data);
	data.dump();

	lite_hdf5_read("demo.h5", "label", 0, 10, label.data);
	label.dump();
	return 0;
}
#endif

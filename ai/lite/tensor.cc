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
	Dtype* data = nullptr;

	// Empty tensor constructor
	Tensor(void) {
		data = nullptr;
	}

	// 1D (vector) constructor
	Tensor(size_t length) {
		this->shape.push_back(length);
		this->initMem();
	}

	// 2D (matrix) constructor
	Tensor(size_t row, size_t col) {
		this->shape.push_back(row);
		this->shape.push_back(col);
		this->initMem();
	}

	// 1D data pointer
	Dtype* at(size_t row, size_t col) {
		return this->data + row*this->shape[1] + col;
	}

	// 2D data pointer
	Dtype* at(size_t offset) {
		return this->data + offset;
	}

	// common dump
	void dump() {
		if (shape.size() == 0) {
			std::cout << "[ ]" << std::endl;
		} else if (shape.size() == 1) {
			std::cout << "[";
			for (int i = 0; i < this->getSize(0); i++)
				printf(" %.3f", *this->at(i));
			std::cout << " ]" << std::endl;
		} else if (shape.size() == 2) {
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
	}

	// common get dimension
	size_t getDim() const {
		return shape.size();
	}

	// common get size
	size_t getSize() const {
		if (shape.empty()) return 0;
		size_t size = 1;
		for (auto i: shape) size *= i;
		return size;
	}

	// common get size
	size_t getSize(int i) const {
		return (i >= shape.size()) ? -1 : shape[i];
	}

	// common init
	void initMem() {
		assert(data == nullptr);
		data = (Dtype*)malloc(sizeof(Dtype)*getSize());
		memset(data, 0x0, sizeof(Dtype)*getSize());
	}

	// common resize to 1D
	Tensor<Dtype>* resize(size_t length) {
		shape.clear();
		if (data != nullptr) free(data);
		data = nullptr;
		shape.push_back(length);
		initMem();
		return this;
	}

	// common resize to 2D
	Tensor<Dtype>* resize(size_t row, size_t col) {
		shape.clear();
		if (data != nullptr) free(data);
		data = nullptr;
		shape.push_back(row);
		shape.push_back(col);
		initMem();
		return this;
	}

	// common inplace zero
	Tensor<Dtype>* zero_() {
		memset(data, 0x0, sizeof(Dtype)*getSize());
		return this;
	}

	// common inplace fill
	Tensor<Dtype>* fill_(Dtype value) {
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = (Dtype) value;
		return this;
	}

	// common rand ~U(0,1)
	Tensor<Dtype>* rand_(void) {
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = (Dtype)random()/RAND_MAX;
	}

	// 2D transpose, non-inplace
	Tensor<Dtype>* transpose(void) {
		assert(shape.size() == 2);
		auto xT = new Tensor<Dtype> (shape[1], shape[0]);
		for (int i = 0; i < shape[0]; i++)
			for (int j = 0; j < shape[1]; j++)
				*xT->at(j, i) = *at(i, j);
		return xT;
	}
};

// LEVEL1 BLAS: AXPY : Y <- aX + Y
template <typename Dtype>
void
AXPY(Dtype alpha, Tensor<Dtype>* X, Tensor<Dtype>* Y)
{
	// regard tensor as a flattened
	assert(X->getSize() == Y->getSize());
	for (size_t i = 0; i < X->getSize(); i++)
		*Y->at(i) += alpha * *X->at(i);
}

// LEVEL3 BLAS: GEMM : C <- aAB + bC
template <typename Dtype>
void
GEMM(Dtype alpha, Tensor<Dtype>* A, Tensor<Dtype>* B,
		Dtype beta, Tensor<Dtype>* C)
{
	if (A->shape[1] != B->shape[0] || A->shape[0] != C->shape[0] || B->shape[1] != C->shape[1]) {
		fprintf(stderr, "GEMM: Illegal Shape! (%d,%d)x(%d,%d)->(%d,%d)",
				A->shape[0], A->shape[1], B->shape[0], B->shape[1],
				C->shape[0], C->shape[1]);
	}
	// check shape
	assert(A->shape[1] == B->shape[0]);
	assert(A->shape[0] == C->shape[0] && B->shape[1] == C->shape[1]);
	// do gemm
	for (int i = 0; i < C->shape[0]; i++) {
		for (int j = 0; j < C->shape[1]; j++) {
			*C->at(i, j) *= beta;
			for (int k = 0; k < A->shape[1]; k++) {
				*C->at(i, j) += alpha * *A->at(i, k) * *B->at(k, j);
			}
		}
	}
}

#if defined(LITE_TEST_TENSOR)
#include "dataloader.cc"
int
main(void)
{
	Tensor<double> data (10, 784);
	Tensor<double> label (10);

	lite_hdf5_read("demo.h5", "data", 0, 0, 10, 784, data.data);
	data.dump();

	lite_hdf5_read("demo.h5", "label", 0, 10, label.data);
	label.dump();

	Tensor<double> empty;
	empty.dump();
	empty.resize(10);
	empty.dump();
	empty.resize(10, 10);
	empty.dump();
	Tensor<double> ones;
	ones.resize(10, 10)->fill_(1.)->dump();
	GEMM(1., &ones, &ones, 0., &empty);
	empty.dump();

	empty.resize(5, 10);
	empty.rand_();
	empty.dump();
	empty.transpose()->dump();

	AXPY(1., &empty, &empty);
	empty.dump();
	return 0;
}
#endif

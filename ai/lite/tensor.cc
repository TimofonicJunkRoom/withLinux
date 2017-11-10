#if !defined(_LITE_TENSOR_CC)
#define _LITE_TENSOR_CC

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

	// 1D (vector) constructor from raw data
	Tensor(Dtype* mem, size_t length) {
		shape.push_back(length);
		data = mem;
	}

	// 2D (matrix) constructor
	Tensor(size_t row, size_t col) {
		this->shape.push_back(row);
		this->shape.push_back(col);
		this->initMem();
	}

	// 2D (matrix) constructor from raw data
	Tensor(Dtype* mem, size_t row, size_t col) {
		shape.push_back(row);
		shape.push_back(col);
		data = mem;
	}

	// 1D data pointer
	Dtype* at(size_t row, size_t col) {
		return this->data + row*this->shape[1] + col;
	}

	// 2D data pointer
	Dtype* at(size_t offset) {
		return this->data + offset;
	}

	// 2D subTensor, inplace
	Tensor<Dtype>* subTensor_(size_t rlower, size_t rupper) {
		if (getDim() == 2) {
			assert(rlower >= 0 && rlower < shape[0]);
			assert(rupper >= 0 && rupper < shape[0]);
			assert(rlower <= rupper);
			return new Tensor<Dtype>(data+rlower*shape[1], rupper-rlower, shape[1]);
		} else if (getDim() == 1) {
			assert(rlower >= 0 && rlower < getSize());
			assert(rupper >= 0 && rupper < getSize());
			assert(rlower <= rupper);
			return new Tensor<Dtype>(data+rlower, rupper-rlower);
		} // else: error
	}

	// common dump
	void dump() {
		if (shape.size() == 0) {
			std::cout << "[ ]" << std::endl << "Tensor(,)" << std::endl;
		} else if (shape.size() == 1) {
			std::cout << "[";
			for (int i = 0; i < this->getSize(0); i++)
				printf(" %.3f", *this->at(i));
			std::cout << " ]" << std::endl;
			std::cout << "Tensor(" << this->getSize(0) << ",)" << std::endl;
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
			std::cout << "Tensor(" << this->getSize(0) <<  "," << this->getSize(1) << ")" << std::endl;
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

	// common resize As
	Tensor<Dtype>* resizeAs(Tensor<Dtype>* x) {
		if (x->getDim() == 1) {
			this->resize(x->getSize(0));
		} else if (x->getDim() == 2) {
			this->resize(x->getSize(0), x->getSize(1));
		}
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

	// common inplace scal
	Tensor<Dtype>* scal_(Dtype factor) {
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) *= factor;
		return this;
	}

	// common rand ~U(0,1)
	Tensor<Dtype>* rand_(void) {
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = (Dtype)random()/RAND_MAX;
	}

	// common element add, inplace
	void add_(Dtype constant) {
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) += constant;
	}

	// common clone
	Tensor<Dtype>* clone(void) {
		auto y = new Tensor<Dtype> ();
		y->resizeAs(this);
		memcpy(y->data, this->data, sizeof(Dtype)*this->getSize());
		return y;
	}

	// 2D transpose, non-inplace
	Tensor<Dtype>* transpose(void) {
		if (shape.size() != 2) cout << "transpose(): ERROR: shape.size = " << shape.size() << endl;
		assert(shape.size() == 2);
		auto xT = new Tensor<Dtype> ((size_t)shape[1], (size_t)shape[0]);
		for (int i = 0; i < shape[0]; i++)
			for (int j = 0; j < shape[1]; j++)
				*xT->at(j, i) = *at(i, j);
		return xT;
	}

	// common exp
	Tensor<Dtype>* exp(void) {
		auto y = this->clone();
		for (size_t i = 0; i < getSize(); i++)
			*y->at(i) = std::exp(*this->at(i));
		return y;
	}

	// common asum
	Dtype asum(void) {
		Dtype ret = 0.;
		for (size_t i = 0; i < getSize(); i++)
			ret += *at(i) > 0. ? *at(i) : -*at(i);
		return ret;
	}

	// common sum
	Dtype sum(void) {
		Dtype ret = 0.;
		for (size_t i = 0; i < getSize(); i++)
			ret += *at(i);
		return ret;
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
#endif

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

	cout << "clone" << endl;
	auto xxx = empty.clone();
	xxx->dump();

	auto xxx2 = xxx->transpose();
	xxx2->dump();

	auto xxx2sub = xxx2->subTensor_(2,5);
	xxx2sub->dump();

	return 0;
}
#endif

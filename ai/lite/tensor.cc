#if !defined(_LITE_TENSOR_H)
#define _LITE_TENSOR_H

#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <string>

#if defined(USE_OPENMP)
#include <omp.h>
#endif

using namespace std;

template <typename Dtype>
class Tensor {
public:

	// name of the tensor, optional
	std::string name;

	// tensor shape, shape.size() = tensor dimension
	std::vector<size_t> shape;

	// dynamic linear memory block for the tensor
	Dtype* data = nullptr;

	// common destructor
	~Tensor(void) {
		shape.clear();
		if (nullptr != data) free(data);
	}

	// common setName
	void setName(string name) {
		this->name = name;
	}

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
		this->shape.push_back(length);
		this->initMem();
		memcpy(data, mem, sizeof(Dtype)*length);
	}

	// 2D (matrix) constructor
	Tensor(size_t row, size_t col) {
		this->shape.push_back(row);
		this->shape.push_back(col);
		this->initMem();
	}

	// 2D (matrix) constructor from raw data
	Tensor(Dtype* mem, size_t row, size_t col) {
		this->shape.push_back(row);
		this->shape.push_back(col);
		this->initMem();
		memcpy(data, mem, sizeof(Dtype)*row*col);
	}

	// 1D data locator
	Dtype* at(size_t offset) {
		return this->data + offset;
	}

	// 2D data locator
	Dtype* at(size_t row, size_t col) {
		return this->data + row*this->shape[1] + col;
	}

	// data copier
	void copy(Dtype* mem, size_t sz) {
		assert(sz <= getSize());
		memcpy(data, mem, sizeof(Dtype)*sz);
	}

	// 2D slice of rows, XXX: don't forget to delete
	Tensor<Dtype>* sliceRows(size_t rlower, size_t rupper) {
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
		} else {
			fprintf(stderr, "subTensor_: Invalid Instance.\n");
			exit(EXIT_FAILURE);
		}
	}

	// common dump
	void dump() {
		if (shape.size() == 0) {
			std::cout << "[ ]" << std::endl << "Tensor(,)" << std::endl;
			std::cout << "Tensor of name \"" << name << "\", shape (,)"
			   << std::endl;
		} else if (shape.size() == 1) {
			std::cout << "[";
			for (size_t i = 0; i < this->getSize(0); i++)
				printf(" %.3f", *this->at(i));
			std::cout << " ]" << std::endl;
			std::cout << "Tensor of name \"" << name << "\", shape ("
			   << this->getSize(0) <<  ",)" << std::endl;
		} else if (shape.size() == 2) {
			std::cout << "[" << std::endl;;
			for (size_t i = 0; i < this->getSize(0); i++) {
				std::cout << "  [";
				for (size_t j = 0; j < this->getSize(1); j++) {
					printf(" %.3f", *this->at(i, j));
				}
				std::cout << " ]" << std::endl;
			}
			std::cout << "]" << std::endl;
			std::cout << "Tensor of name \"" << name << "\", shape ("
			   << this->getSize(0) <<  "," << this->getSize(1) << ")"
			   << std::endl;
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
	size_t getSize(size_t i) const {
		return (i >= shape.size()) ? -1 : shape[i];
	}

	// common init
	void initMem() {
		if (data != nullptr) free(data);
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
		return this;
	}

	// common inplace zero
	Tensor<Dtype>* zero_() {
		memset(data, 0x0, sizeof(Dtype)*getSize());
		return this;
	}

	// common inplace fill
	Tensor<Dtype>* fill_(Dtype value) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data,value)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = (Dtype) value;
		return this;
	}

	// common inplace scal
	Tensor<Dtype>* scal_(Dtype factor) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data,factor)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) *= factor;
		return this;
	}

	// common rand ~U(0,1)
	Tensor<Dtype>* rand_(void) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = (Dtype)random()/RAND_MAX;
		return this;
	}

	// common uniford ~U(l, u)
	Tensor<Dtype>* uniform(Dtype l, Dtype u) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(l, u)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = ((Dtype)random() / RAND_MAX) * (u-l) + l;
		return this;
	}

	// common element add, inplace
	void add_(Dtype constant) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data,constant)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) += constant;
	}

	// common clone, XXX: don't forget to delete
	Tensor<Dtype>* clone(void) {
		Tensor<Dtype>* y = new Tensor<Dtype> ();
		y->resizeAs(this);
		memcpy(y->data, this->data, sizeof(Dtype)*this->getSize());
		return y;
	}

	// 2D transpose, non-inplace, XXX: don't forget to delete
	Tensor<Dtype>* transpose(void) {
		if (shape.size() != 2) {
			fprintf(stderr, "transpose(): ERROR: shape.size = %ld\n", shape.size());
			exit(EXIT_FAILURE);
		}
		auto xT = new Tensor<Dtype> ((size_t)shape[1], (size_t)shape[0]);
		for (size_t i = 0; i < shape[0]; i++)
			for (size_t j = 0; j < shape[1]; j++)
				*xT->at(j, i) = *at(i, j);
		return xT;
	}

	// common exp, inplace
	Tensor<Dtype>* exp_(void) {
#if defined(USE_OPENMP)
#pragma omp parallel for shared(data)
#endif
		for (size_t i = 0; i < getSize(); i++)
			*(data + i) = std::exp(*(data + i));
		return this;
	}

	// common asum
	Dtype asum(void) {
		Dtype ret = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret)
#endif
		for (size_t i = 0; i < getSize(); i++)
			ret += *at(i) > 0. ? *at(i) : -*at(i);
		return ret;
	}

	// common sum
	Dtype sum(void) {
		Dtype ret = 0.;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret)
#endif
		for (size_t i = 0; i < getSize(); i++)
			ret += *at(i);
		return ret;
	}

	// compares size of two tensors
	bool sameSize(Tensor<Dtype>* x) {
		if (x->getDim() != getDim()) return false;
		for (size_t i = 0; i < shape.size(); i++)
			if (x->getSize(i) != getSize(i)) return false;
		return true;
	}
};

// common dump by overloading operator
template <typename Dtype>
std::ostream& operator<< (std::ostream& out, Tensor<Dtype>& x) {
	x.dump();
	return out;
}

// LEVEL1 BLAS: AXPY : Y <- aX + Y
template <typename Dtype>
void
AXPY(Dtype alpha, Tensor<Dtype>* X, Tensor<Dtype>* Y)
{
	// regard tensor as a flattened
	assert(X->getSize() == Y->getSize());
#if defined(USE_OPENMP)
#pragma omp parallel for shared(X, Y)
#endif
	for (size_t i = 0; i < X->getSize(); i++)
		*Y->at(i) += alpha * *X->at(i);
}

// LEVEL3 BLAS: GEMM : C <- aAB + bC
template <typename Dtype>
void
GEMM(Dtype alpha, Tensor<Dtype>* A, Tensor<Dtype>* B,
		Dtype beta, Tensor<Dtype>* C)
{
	// check shape
	if (A->shape[1] != B->shape[0] || A->shape[0] != C->shape[0] || B->shape[1] != C->shape[1]) {
		fprintf(stderr, "GEMM: Illegal Shape! (%ld,%ld)x(%ld,%ld)->(%ld,%ld)",
				A->shape[0], A->shape[1], B->shape[0], B->shape[1],
				C->shape[0], C->shape[1]);
		exit(EXIT_FAILURE);
	}
	size_t i = 0, j = 0, k = 0;
#if defined(USE_OPENMP)
#pragma omp parallel for collapse(2) shared(A, B, C) private(k)
#endif
	for (i = 0; i < C->shape[0]; i++) {
		for (j = 0; j < C->shape[1]; j++) {
			*C->at(i, j) *= beta;
			for (k = 0; k < A->shape[1]; k++) {
				*C->at(i, j) += alpha * *A->at(i, k) * *B->at(k, j);
			}
		}
	}
}

// MAE: y <- sum_i ||a_i - b_i||_1
template <typename Dtype>
Dtype
MAE(Tensor<Dtype>* A, Tensor<Dtype>* B) {
	Dtype ret = 0.;
	size_t minsize = A->getSize();
	minsize = (B->getSize() < minsize) ? B->getSize() : minsize;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret) shared(minsize)
#endif
	for (size_t i = 0; i < minsize; i++) {
		Dtype tmp = *A->at(i) - *B->at(i);
		ret += (tmp > (Dtype)0.) ? tmp : -tmp;
	}
	return ret;
}

// MSE: y <- sum_i ||a_i - b_i||_2^2
template <typename Dtype>
Dtype
MSE(Tensor<Dtype>* A, Tensor<Dtype>* B) {
	Dtype ret = 0.;
	size_t minsize = A->getSize();
	minsize = (B->getSize() < minsize) ? B->getSize() : minsize;
#if defined(USE_OPENMP)
#pragma omp parallel for reduction (+:ret) shared(minsize)
#endif
	for (size_t i = 0; i < minsize; i++) {
		Dtype tmp = *A->at(i) - *B->at(i);
		ret += tmp * tmp;
	}
	return ret;
}

#endif // defined(_LITE_TENSOR_H)

#if defined(LITE_TEST_TENSOR)
#include "dataloader.cc"

string _msg_;
inline string _padding_(string msg) {
	for (size_t i = 0; i < 80-msg.size(); i++) cout << " ";
	return msg;
}
#define TS(msg) do { \
  _msg_ = msg; \
  cout << endl << "... " << _padding_(_msg_) << " [ .. ]" << endl; \
 } while (0)
#define TE do { \
  cout << ">>> " << _padding_(_msg_) << " [ OK ]" << endl; \
 } while (0)

int
main(void)
{
	cout << "::         " << _padding_("Tensor Tests") << endl;

	TS("tensor creation"); {
		Tensor<double> matrix (10, 10);
		Tensor<double> vector (10);
		matrix.dump();
		vector.dump();
	}; TE;

	TS("feed hdf5 data to tensor"); {
		Tensor<double> matrix (10, 784);
		Tensor<double> vector (10);
		vector.name = "label";
		lite_hdf5_read("demo.h5", "data", 0, 0, 10, 784, matrix.data);
		lite_hdf5_read("demo.h5", "label", 0, 10, vector.data);
		vector.dump();
	}; TE;

	TS("resize from empty tensor"); {
		Tensor<double> empty;
		empty.resize(10);
		empty.dump();
		empty.resize(10, 10);
		empty.dump();
		empty.resize(1);
		empty.dump();
	}; TE;

	TS("inplace fill, scal"); {
		Tensor<double> ones;
		ones.resize(10, 10)->fill_(4.2);
		ones.dump();
		ones.scal_(0.5);
		ones.dump();
	}; TE;

	TS("GEMM"); {
		Tensor<double> ones (10, 20);
		ones.fill_(1.);
		ones.name = "ones";
		Tensor<double>* onesT = ones.transpose();
		onesT->name = "onesT";
		Tensor<double> dest (10, 10);
		GEMM(1., &ones, onesT, 0., &dest);
		ones.dump();
		onesT->dump();
		dest.dump();
		delete onesT;
	}; TE;

	TS("inplace random"); {
		Tensor<double> x (5, 10);
		x.rand_();
		x.dump();
	}; TE;

	TS("AXPY"); {
		Tensor<double> x (10, 10);
		x.fill_(2.1);
		AXPY(1., &x, &x);
		x.dump();
	}; TE;

	TS("clone"); {
		Tensor<double> x (10, 10);
		x.rand_();
		x.dump();
		Tensor<double>* y = x.clone();
		y->dump();
		cout << &x << " " << y << endl;
		delete y;
	}; TE;

	TS("sliceRows"); {
		Tensor<double> x (10, 10);
		x.rand_();
		x.dump();
		Tensor<double>* y = x.sliceRows(2, 5);
		y->dump();
		delete y;
	}; TE;

	TS("uniform"); {
		Tensor<double> x (10, 10);
		x.uniform(-10, 10);
		x.dump();
	}; TE;

	TS("<int> sameSize"); {
		Tensor<int> x (10, 10);
		Tensor<int> y (1);
		Tensor<int> z (10, 11);
		assert(x.sameSize(&x) == true);
		assert(x.sameSize(&y) == false);
		assert(x.sameSize(&z) == false);
	}; TE;

	cout << "::         " << _padding_("Tensor Tests OK") << endl;
	return 0;
}
#endif

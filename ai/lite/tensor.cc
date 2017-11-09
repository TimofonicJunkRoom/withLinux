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
	size_t initMem() {
		assert(data == nullptr);
		data = (Dtype*)malloc(sizeof(Dtype)*getSize());
		memset(data, 0x0, sizeof(Dtype)*getSize());
	}

	// common resize to 1D
	void resize(size_t length) {
		shape.clear();
		if (data != nullptr) free(data);
		data = nullptr;
		shape.push_back(length);
		initMem();
	}

	// common resize to 2D
	void resize(size_t row, size_t col) {
		shape.clear();
		if (data != nullptr) free(data);
		data = nullptr;
		shape.push_back(row);
		shape.push_back(col);
		initMem();
	}

	// common inplace zero
	void zero_() {
		memset(data, 0x0, sizeof(Dtype)*getSize());
	}

	// common destructor
	~Tensor() {
		free(this->data);
	}
};

template <typename Dtype>
void
GEMM(Dtype alpha, Tensor<Dtype>* A, Dtype beta, Tensor<Dtype>* B,
		Dtype gamma, Tensor<Dtype>* C)
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
	empty.resize(10, 4);
	empty.dump();
	return 0;
}
#endif

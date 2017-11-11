#if !defined(_LITE_BLOB_H)
#define _LITE_BLOB_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include "tensor.cc"

using namespace std;

template <typename Dtype>
class Blob {
public:
	// holds the value tensor
	Tensor<Dtype> value = Tensor<Dtype>();
	// holds the gradient tensor of value
	Tensor<Dtype> gradient = Tensor<Dtype>();
	// is gradient needed for this blob? true by default
	bool requires_grad = true;
	// optional name
	string name;

	// empty blob constructor
	Blob(){}

	// 1D blob constructor
	Blob(size_t length) {
		this->resize(length);
	}

	// 1D blob constructor, with flag
	Blob(int length, bool requires_grad) {
		this->value.resize(length);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(length);
	}

	// 2D blob constructor
	Blob(int row, int col) {
		this->resize(row, col);
	}

	// 2D blob constructor, with flag
	Blob(int row, int col, bool requires_grad) {
		this->value.resize(row, col);
		this->requires_grad = requires_grad;
		if (requires_grad) this->gradient.resize(row, col);
	}

	// 1D blob resizer
	void resize(int length) {
		value.resize(length);
		if (requires_grad) gradient.resize(length);
	}

	// 2D blob resizer
	void resize(int row, int col) {
		value.resize(row, col);
		if (requires_grad) gradient.resize(row, col);
	}

	// transpose, pseudo-inplace
	void transpose_() {
		assert(value.getDim() == 2);
		Tensor<Dtype>* valueT = value.transpose();
		value.resize(value.shape[1], value.shape[0]);
		value.copy(valueT->data, value.getSize());
		delete valueT;
		if (requires_grad) {
			Tensor<Dtype>* gradientT = gradient.transpose();
			gradient.resize(gradient.shape[1], gradient.shape[0]);
			gradient.copy(gradientT->data, gradient.getSize());
			delete gradientT;
		}
	}

	// blob clone, XXX: don't forget to delete
	Blob<Dtype>* clone() {
		auto newblob = new Blob<Dtype>();
		newblob->name = name;
		newblob->requires_grad = requires_grad;
		newblob->value.resizeAs(&value);
		newblob->gradient.resizeAs(&gradient);
		newblob->value.copy(value.data, value.getSize());
		newblob->gradient.copy(gradient.data, gradient.getSize());
		return newblob;
	}

	// zero gradient
	void zeroGrad() {
		if (requires_grad) this->gradient.zero_();
	}

	// dumper
	void dump() {
		this->value.dump();
		this->gradient.dump();
	}

	// dumper, with flags
	void dump(bool pv, bool pg) {
		if (pv) this->value.dump();
		if (pg) this->gradient.dump();
	}

	// setting name
	void setName(string name) {
		this->name = name;
		this->value.name = name + ".value";
		this->gradient.name = name + ".gradient";
	}
};

#endif // _LITE_BLOB_H

#if defined(LITE_TEST_BLOB)
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
	TS("blob construction and name"); {
		Blob<double> databatch(12, 10); // d=12, batch10
		databatch.setName("databatch");
		databatch.dump();
		Blob<double> databatchnograd(12, 10, false);
		databatchnograd.setName("databatchnograd");
		databatchnograd.dump();
	}; TE;

	TS("blob clone"); {
		Blob<double> x (10, 10);
		x.setName("x");
		x.value.rand_();
		x.gradient.fill_(0.123);
		x.dump();
		Blob<double>* y = x.clone();
		y->setName("y");
		y->value.scal_(2.0);
		y->gradient.scal_(2.0);
		y->dump();
		delete y;
	}; TE;

	TS("blob transpose"); {
		Blob<double> x (10, 10);
		x.value.rand_();
		x.gradient.rand_();
		x.dump();
		x.transpose_();
		x.dump();
	}; TE;

	return 0;
}
#endif

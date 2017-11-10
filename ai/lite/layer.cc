#if !defined(_LITE_LAYER_CC)
#define _LITE_LAYER_CC

#include <iostream>
#include <vector>
#include <algorithm>
#include "tensor.cc"

using namespace std;

template <typename Dtype>
class Blob {
public:
	Tensor<Dtype>* value = new Tensor<Dtype>();
	Tensor<Dtype>* gradient = new Tensor<Dtype>();
	bool requires_grad = true;

	Blob(){}

	Blob(Tensor<Dtype>* value_, Tensor<Dtype>* gradient_, bool requires_grad_) {
		this->value = value_;
		this->gradient = gradient_;
		this->requires_grad = requires_grad_;
	}

	Blob(int length) {
		this->resize(length);
	}

	Blob(int length, bool requires_grad) {
		if (requires_grad) {
			this->resize(length);
		} else {
			this->value->resize(length);
			this->requires_grad = requires_grad;
		}
	}

	Blob(int row, int col, bool requires_grad) {
		if (requires_grad) {
			this->resize(row, col);
		} else {
			this->value->resize(row, col);
			this->requires_grad = requires_grad;
		}
	}

	Blob(int row, int col) {
		this->resize(row, col);
	}

	void resize(int length) {
		value->resize(length);
		if (requires_grad) gradient->resize(length);
	}

	void resize(int row, int col) {
		value->resize(row, col);
		if (requires_grad) gradient->resize(row, col);
	}

	Blob<Dtype>* transpose() {
		assert(value->getDim() == 2);
		auto newx = this->clone();
//		auto oldvalue = value;
		value = value->transpose();
//		auto oldgradient = gradient;
		if (requires_grad) gradient = gradient->transpose();
		return newx;
	}

	Blob<Dtype>* clone() {
		auto newvalue = value->clone();
		auto newgradient = gradient->clone();
		auto newrequires_grad = requires_grad;
		auto newblob = new Blob<Dtype>(newvalue, newgradient, newrequires_grad);
		return newblob;
	}

	void zeroGrad() {
		this->gradient->zero_();
	}

	void dump() {
		this->value->dump();
		this->gradient->dump();
	}

	void dump(bool pv, bool pg) {
		if (pv) this->value->dump();
		if (pg) this->gradient->dump();
	}
};

template <typename Dtype>
class Layer {
public:
	void zeroGrad() {}
	void forward() {}
	void backward() {}
	void update(double lr) {}
	void report() {}
};

template <typename Dtype>
class LinearLayer : public Layer<Dtype> {
public:
	Blob<Dtype> W;
	Blob<Dtype> b;
	bool use_bias = true;

	LinearLayer(int dim_dest, int dim_src) {
		W.resize(dim_dest, dim_src);
		b.resize(dim_dest);
		W.value->rand_();
		W.value->add_(-0.5);
		W.value->scal_(0.001);
		W.gradient->zero_();
		b.value->fill_(0.);
		b.gradient->zero_();
	}

	void zeroGrad(void) {
		W.gradient->zero_();
		b.gradient->zero_();
	}

	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		// output += GEMM(W, X)
		GEMM(1., W.value, input.value, 0., output.value);
		// output += b
		size_t batchsize = input.value->getSize(1);
		size_t outdim = W.value->getSize(0);
		for (size_t j = 0; j < batchsize; j++) {
			for (size_t i = 0; i < outdim; i++) {
				*output.value->at(i, j) += *b.value->at(i);
			}
		}
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		if (!output.requires_grad) return;
		// grad of W: g x x^T
		GEMM(1., output.gradient, input.value->transpose(), 0., W.gradient);
		// grad of X: W^T x g
		if (input.requires_grad)
			GEMM(1., W.gradient->transpose(), output.gradient, 0., input.gradient);
		// grad of b: unexpand(g)
		size_t batchsize = input.value->getSize(1);
		size_t outdim = W.value->getSize(0);
		for (size_t j = 0; j < batchsize; j++) {
			for (size_t i = 0; i < outdim; i++) {
				*b.gradient->at(i) += *output.gradient->at(i, j);
			}
		}
	}

	void update(double lr) {
		AXPY((Dtype)-lr, W.gradient, W.value);
		AXPY((Dtype)-lr, b.gradient, b.value);
	}

	void dumpstat() {
		cout << "> LinearLayer:" << endl;
		cout << "  > W sum " << W.value->sum() << " asum " << W.value->asum() << endl;
		cout << "  > b sum " << b.value->sum() << " asum " << b.value->asum() << endl;
		cout << "  > gradW sum " << W.gradient->sum() << " asum " << W.gradient->asum() << endl;
		cout << "  > gradb sum " << b.gradient->sum() << " asum " << b.gradient->asum() << endl;
	}
};

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {
public:

	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		auto relu = [](Dtype x) { return x > (Dtype)0. ? x : (Dtype)0.; };
		for (size_t i = 0; i < input.value->getSize(); i++)
			*output.value->at(i) = relu(*input.value->at(i));
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t i = 0; i < input.gradient->getSize(); i++) {
			if (*input.value->at(i) > (Dtype)0.)
				*input.gradient->at(i) = *output.gradient->at(i);
			else
				*input.gradient->at(i) = (Dtype)0.;
		}
	}
};

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		// input.exp().sum(0), sum in the first row
		auto expx = input.value->exp();
		for (size_t i = 1; i < expx->getSize(0); i++)
			for (size_t j = 0; j < expx->getSize(1); j++)
				*expx->at(0, j) += *expx->at(i, j);
		// output
		for (size_t i = 0; i < expx->getSize(0); i++)
			for (size_t j = 0; j < expx->getSize(1); j++)
				*output.value->at(i, j) = std::exp(*input.value->at(i,j)) /
					((Dtype)1e-7 + *expx->at(0, j));
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t sample = 0; sample < input.gradient->getSize(1); sample++) {
			for (size_t row = 0; row < input.gradient->getSize(0); row++) {
				Dtype element = 0.;
				for (size_t k = 0; k < output.value->getSize(0); k++) {
					element -= (*output.gradient->at(k, sample)) *
						(*output.value->at(k,sample) * *output.value->at(row, sample));
					if (k == row)
						element += (*output.gradient->at(k, sample)) *
							(*output.value->at(row,sample) * *output.value->at(row, sample));
				}
				*input.gradient->at(row, sample) = element;
			}
		}
	}
};

template <typename Dtype>
class MSELoss : public Layer<Dtype> {
public:
	double lossval = 0.;
	double MAE = 0.;

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype> label) {
		lossval = 0.;
		MAE = 0.;
		size_t numsamples = input.value->getSize(1);
		size_t numdim = input.value->getSize(0);
		auto square = [](Dtype x) { return x*x; };
		auto myabs  = [](Dtype x) { return x>0?x:-x; };
		for (size_t i = 0; i < numsamples; i++) {
			for (size_t j = 0; j < numdim; j++) {
				lossval += square(*input.value->at(i,j) - *label.value->at(i,j));
				MAE     += myabs(*input.value->at(i,j) - *label.value->at(i,j));
			}
		}
		lossval /= numsamples;
		*output.value->at(0) = lossval;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype> label) {
		size_t numsamples = input.value->getSize(1);
		input.gradient->zero_();
		AXPY(1., input.value, input.gradient);
		AXPY(-1., label.value, input.gradient);
		input.gradient->scal_(2./numsamples);
	}

	void report() {
		std::cout << " * MSELoss: " << lossval << " (MAE " << MAE << ")" << std::endl;
	}
};

template <typename Dtype>
class ClassNLLLoss : public Layer<Dtype> {
public:
	double lossval = 0.;

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype> label) {
		lossval = 0.;
		size_t samples = input.value->getSize(1);
		for (size_t i = 0; i < samples; i++)
			lossval += - log(1e-7 + *input.value->at((size_t)*label.value->at(i), i));
		lossval /= samples;
		*output.value->at(0) = (Dtype)lossval;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype> label) {
		input.gradient->zero_();
		size_t samples = input.value->getSize(0);
		for (size_t i = 0; i < samples; i++)
			*input.gradient->at(*label.value->at(i), i) =
				- 1. / (1e-7 + *input.value->at((size_t)*label.value->at(i), i));
	}

	void report() {
		std::cout << " * ClassNLLLoss: " << lossval << std::endl;
	}
};

template <typename Dtype>
class ClassAccuracy : public Layer<Dtype> {
public:
	double accuracy = 0.;
	size_t numsamples = 0;
	size_t numcorrect = 0;
	size_t numclass = 0;

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype> label) {
		numsamples = input.value->getSize(1);
		numclass   = input.value->getSize(0);
		numcorrect = 0;
		for (size_t j = 0; j < numsamples; j++) {
			bool dirty = false;
			for (size_t i = 0; i < numclass; i++) {
				if ((size_t)*label.value->at(j) == i) continue;
				if (*input.value->at((size_t)*label.value->at(j), j)
				  <= *input.value->at(i, j)) {
					dirty = true;
					break;
				}
			}
			if (!dirty) numcorrect++;
		}
		accuracy = (double)numcorrect / numsamples;
		*output.value->at(0) = accuracy;
	}

	void report() {
		std::cout << " * Accuracy: " << accuracy << " (" << numcorrect << "/" << numsamples << ")" << std::endl;
	}
};
#endif

#if defined(LITE_TEST_LAYER)
#include "dataloader.cc"
int
main(void)
{
	Blob<double> databatch(12, 10); // d=12, batch10
	Blob<double> out1 (3, 10);
	databatch.value->fill_(1.);
	//databatch.value->rand_();
	cout << "databatch"; databatch.value->dump();

	auto fc1 = LinearLayer<double>(3, 12);
	fc1.W.value->fill_(1.);
	fc1.b.value->fill_(.1);
	cout << "W"; fc1.W.value->dump();

	fc1.forward(databatch, out1);
	out1.value->dump();
	fc1.backward(databatch, out1);

	cout << "clone" << endl;
	auto xxx = databatch.clone();
	xxx->dump();
	cout << "transpose" << endl;
	xxx->transpose();
	xxx->dump();
	xxx->transpose();

	cout << "softmax" << endl;
	auto sm1 = SoftmaxLayer<double> ();
	sm1.forward(databatch, *xxx);
	databatch.dump();
	xxx->dump();
	cout << "softmax back" << endl;
	//xxx->gradient->fill_(1.);
	xxx->gradient->rand_();
	databatch.zeroGrad();
	sm1.backward(databatch, *xxx);
	xxx->dump();
	databatch.dump();

	return 0;
}
#endif

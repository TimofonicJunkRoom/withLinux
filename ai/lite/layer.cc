#if !defined(_LITE_LAYER_CC)
#define _LITE_LAYER_CC

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "tensor.cc"
#include "blob.cc"

using namespace std;

// Basic Layer Class
template <typename Dtype>
class Layer {
public:
	void zeroGrad() {}
	void forward() {}
	void backward() {}
	void update(double lr) {}
};

// Linear, y <- Wx + b
template <typename Dtype>
class LinearLayer : public Layer<Dtype> {
public:
	Blob<Dtype> W;
	Blob<Dtype> b;
	bool use_bias = true;

	LinearLayer(int dim_dest, int dim_src, bool use_bias=true) {
		W.resize(dim_dest, dim_src);
		b.resize(dim_dest);
		W.setName("LinearLayer/W");
		b.setName("LinearLayer/b");
		W.gradient.zero_();
		b.gradient.zero_();
		// parameter initialization
		// ref Torch:nn, W,b ~ uniform(-stdv, stdv)
		//     where stdv = 1. / sqrt(inputSize)
		this->use_bias = use_bias;
		double stdv = 1. / std::sqrt(dim_src);
		W.value.uniform(-stdv, stdv);
		if (use_bias) b.value.uniform(-stdv, stdv);
	}

	void zeroGrad(void) {
		W.gradient.zero_();
		b.gradient.zero_();
	}

	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		// output += GEMM(W, X)
		GEMM(1., &W.value, &input.value, 0., &output.value);
		// output += b
		size_t batchsize = input.value.getSize(1);
		size_t outdim = W.value.getSize(0);
		for (size_t j = 0; j < batchsize; j++) {
			for (size_t i = 0; i < outdim; i++) {
				*output.value.at(i, j) += *b.value.at(i);
			}
		}
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		if (!output.requires_grad) return;
		auto inputvalueT = input.value.transpose();
		auto wvalueT = W.value.transpose();
		// grad of W: g x x^T
		GEMM(1., &output.gradient, inputvalueT, 0., &W.gradient);
		// grad of X: W^T x g
		if (input.requires_grad) {
			GEMM(1., wvalueT, &output.gradient, 0., &input.gradient);
		}
		// grad of b: unexpand(g)
		size_t batchsize = input.value.getSize(1);
		size_t outdim = W.value.getSize(0);
		for (size_t j = 0; j < batchsize; j++) {
			for (size_t i = 0; i < outdim; i++) {
				*b.gradient.at(i) += *output.gradient.at(i, j);
			}
		}
		delete inputvalueT;
		delete wvalueT;
	}

	void SGD(double lr) {
		AXPY((Dtype)-lr, &W.gradient, &W.value);
		AXPY((Dtype)-lr, &b.gradient, &b.value);
	}

	void dumpstat() {
		cout << "  > LinearLayer:" << endl;
		cout << "    * W sum " << W.value->sum() << "\tasum " << W.value->asum();
		cout << "\t | gradW sum " << W.gradient->sum() << "\tasum " << W.gradient->asum() << endl;
		cout << "    * b sum " << b.value->sum() << "\tasum " << b.value->asum();
		cout << "\t | gradb sum " << b.gradient->sum() << "\tasum " << b.gradient->asum() << endl;
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
		MAE     /= numsamples;
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
		std::cout << " * MSELoss: " << lossval << "\t(MAE " << MAE << ")" << std::endl;
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
	TS("linear layer"); {
		// prepare
		Blob<double> X (4, 5); // sample=10, inputSize=12
		X.setName("X");
		X.value.rand_();
		X.dump(true, false);
		Blob<double> yhat (2, 5);
		yhat.setName("yhat");
		LinearLayer<double> fc1 (2, 4);
		fc1.W.dump(true, false);
		fc1.b.dump(true, false);
		// forward
		fc1.forward(X, yhat);
		yhat.gradient.fill_(1.);
		yhat.dump();
		// backward
		fc1.backward(X, yhat);
		fc1.W.dump();
		fc1.b.dump();
		X.dump();
		// update
		fc1.SGD(1e-3);
	}; TE;

	//auto fc1 = LinearLayer<double>(3, 12);
	//fc1.W.value->fill_(1.);
	//fc1.b.value->fill_(.1);
	//cout << "W"; fc1.W.value->dump();

	//fc1.forward(databatch, out1);
	//out1.value->dump();
	//fc1.backward(databatch, out1);

	//cout << "clone" << endl;
	//auto xxx = databatch.clone();
	//xxx->dump();
	//cout << "transpose" << endl;
	//xxx->transpose();
	//xxx->dump();
	//xxx->transpose();

	//cout << "softmax" << endl;
	//auto sm1 = SoftmaxLayer<double> ();
	//sm1.forward(databatch, *xxx);
	//databatch.dump();
	//xxx->dump();
	//cout << "softmax back" << endl;
	////xxx->gradient->fill_(1.);
	//xxx->gradient->rand_();
	//databatch.zeroGrad();
	//sm1.backward(databatch, *xxx);
	//xxx->dump();
	//databatch.dump();

	return 0;
}
#endif

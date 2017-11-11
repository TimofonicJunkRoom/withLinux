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
		this->use_bias = use_bias;
		// setup this layer
		W.resize(dim_dest, dim_src);
		if (use_bias) b.resize(dim_dest);
		W.setName("LinearLayer/W");
		if (use_bias) b.setName("LinearLayer/b");
		W.gradient.zero_();
		if (use_bias) b.gradient.zero_();
		// parameter initialization
		// ref Torch:nn, W,b ~ uniform(-stdv, stdv)
		//     where stdv = 1. / sqrt(inputSize)
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
		if (use_bias) {
			size_t batchsize = input.value.getSize(1);
			size_t outdim = W.value.getSize(0);
			for (size_t j = 0; j < batchsize; j++) {
				for (size_t i = 0; i < outdim; i++) {
					*output.value.at(i, j) += *b.value.at(i);
				}
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
		if (use_bias) {
			size_t batchsize = input.value.getSize(1);
			size_t outdim = W.value.getSize(0);
			for (size_t j = 0; j < batchsize; j++) {
				for (size_t i = 0; i < outdim; i++) {
					*b.gradient.at(i) += *output.gradient.at(i, j);
				}
			}
		}
		delete inputvalueT;
		delete wvalueT;
	}

	void update(double lr) {
		SGD(lr);
	}

	void SGD(double lr) {
		AXPY((Dtype)-lr, &W.gradient, &W.value);
		AXPY((Dtype)-lr, &b.gradient, &b.value);
	}

	void dumpstat() {
		cout << "  > LinearLayer:" << endl;
		cout << "    * W sum " << W.value.sum() << "\tasum " << W.value.asum();
		cout << "\t | gradW sum " << W.gradient.sum() << "\tasum " << W.gradient.asum() << endl;
		cout << "    * b sum " << b.value.sum() << "\tasum " << b.value.asum();
		cout << "\t | gradb sum " << b.gradient.sum() << "\tasum " << b.gradient.asum() << endl;
	}
};

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {
public:

	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		auto relu = [](Dtype x) { return x > (Dtype)0. ? x : (Dtype)0.; };
		for (size_t i = 0; i < input.value.getSize(); i++)
			*output.value.at(i) = relu(*input.value.at(i));
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t i = 0; i < input.gradient.getSize(); i++) {
			if (*input.value.at(i) > (Dtype)0.)
				*input.gradient.at(i) = *output.gradient.at(i);
			else
				*input.gradient.at(i) = (Dtype)0.;
		}
	}
};

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
	void forward(Blob<Dtype>& input, Blob<Dtype>& output) {
		// input.exp().sum(0), sum in the first row
		Tensor<Dtype>* expx = input.value.clone();
		for (size_t j = 0; j < expx->getSize(1); j++) {
			// find maxval of this colomn
			Dtype maxval = *expx->at(0, j);
			for (size_t i = 0; i < expx->getSize(0); i++)
				if (maxval < *expx->at(i,j)) maxval = *expx->at(i,j);
			// subtract the maxval from this column
			for (size_t i = 0; i < expx->getSize(0); i++)
				*expx->at(i,j) -= maxval;
		}
		expx->exp_();
		// save the exp(x_ij) result to output
		output.value.copy(expx->data, output.value.getSize());
		// sum up each column
		for (size_t i = 1; i < expx->getSize(0); i++)
			for (size_t j = 0; j < expx->getSize(1); j++)
				*expx->at(0, j) += *expx->at(i, j);
		// output
		for (size_t i = 0; i < expx->getSize(0); i++)
			for (size_t j = 0; j < expx->getSize(1); j++)
				*output.value.at(i, j) /= (Dtype)1e-7 + *expx->at(0, j);
		delete expx;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output) {
		for (size_t sample = 0; sample < input.gradient.getSize(1); sample++) {
			for (size_t row = 0; row < input.gradient.getSize(0); row++) {
				Dtype element = 0.;
				for (size_t k = 0; k < output.gradient.getSize(0); k++) {
					element -= (*output.gradient.at(k, sample))
						* (*output.value.at(row, sample))
						* (*output.value.at(k,sample));
					if (k == row)
						element += (*output.gradient.at(k, sample))
							* (*output.value.at(row,sample));
				}
				*input.gradient.at(row, sample) = element;
			}
		}
	}
};

template <typename Dtype>
class MSELoss : public Layer<Dtype> {
public:
	double lossval = 0.;
	double MAE = 0.;

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		if (!label.sameSize(&input)) {
			fprintf(stderr, "MSELoss: input and GT size differs!\n");
			exit(EXIT_FAILURE);
		}
		lossval = 0.;
		MAE = 0.;
		size_t numsamples = input.value.getSize(1);
		size_t numdim = input.value.getSize(0);
		auto square = [](Dtype x) { return x*x; };
		auto myabs  = [](Dtype x) { return x>0?x:-x; };
		for (size_t i = 0; i < numsamples; i++) {
			for (size_t j = 0; j < numdim; j++) {
				lossval += square(*input.value.at(i,j) - *label.value.at(i,j));
				MAE     += myabs(*input.value.at(i,j) - *label.value.at(i,j));
			}
		}
		lossval /= numsamples;
		MAE     /= numsamples;
		*output.value.at(0) = lossval;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		size_t numsamples = input.value.getSize(1);
		input.gradient.zero_();
		AXPY(1., &input.value, &input.gradient);
		AXPY(-1., &label.value, &input.gradient);
		input.gradient.scal_(2./numsamples);
	}

	void report() {
		std::cout << " * MSELoss: " << lossval << "\t(MAE " << MAE << ")" << std::endl;
	}
};

template <typename Dtype>
class ClassNLLLoss : public Layer<Dtype> {
public:
	double lossval = 0.;

	bool _checksize(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		if (label.value.getDim() == 1) {
			if (input.value.shape[1] != label.value.getSize()) return false;
		} else if (label.value.getDim() == 2) {
			if (input.value.shape[1] != label.value.shape[1]) return false;
		}
		return true;
	}

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		assert(true == _checksize(input, output, label));
		lossval = 0.;
		size_t samples = input.value.getSize(1);
		for (size_t i = 0; i < samples; i++)
			lossval += - log(1e-7 + *input.value.at((size_t)*label.value.at(i), i));
		lossval /= samples;
		*output.value.at(0) = (Dtype)lossval;
	}

	void backward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		assert(true == _checksize(input, output, label));
		input.gradient.zero_();
		size_t samples = input.value.getSize(1);
		for (size_t i = 0; i < samples; i++)
			*input.gradient.at(*label.value.at(i), i) =
				- 1. / (1e-7 + *input.value.at((size_t)*label.value.at(i), i));
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

	void forward(Blob<Dtype>& input, Blob<Dtype>& output, Blob<Dtype>& label) {
		numsamples = input.value.getSize(1);
		numclass   = input.value.getSize(0);
		numcorrect = 0;
		for (size_t j = 0; j < numsamples; j++) {
			bool dirty = false;
			for (size_t i = 0; i < numclass; i++) {
				size_t locj = (size_t)*label.value.at(j);
				if (locj == i) continue;
				if (*input.value.at(i, j) <= *input.value.at(locj, j)) {
					dirty = true;
					break;
				}
			}
			if (!dirty) numcorrect++;
		}
		accuracy = (double)numcorrect / numsamples;
		*output.value.at(0) = accuracy;
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
		// without bias
		LinearLayer<double> fc2 (2, 4, false);
		fc2.forward(X, yhat);
		fc2.backward(X, yhat);
	}; TE;

	TS("relu layer"); {
		Blob<double> X (5, 10);
		X.setName("X");
		X.value.rand_()->add_(-.5);
		X.gradient.fill_(1.);
		X.dump();
		ReluLayer<double> relu1;
		relu1.forward(X, X);
		relu1.backward(X, X);
		X.dump();
	}; TE;

	TS("MSE layer"); {
		Blob<double> y (10, 1);
		y.setName("y");
		y.value.fill_(0.);
		Blob<double> yhat(10, 1);
		yhat.setName("yhat");
		yhat.value.fill_(1.);
		Blob<double> loss (1);
		MSELoss<double> loss1;
		loss1.forward(yhat, loss, y);
		loss1.report();
		loss1.backward(yhat, loss, y);
		y.dump();
		yhat.dump();
	}; TE;

	TS("softmax layer"); {
		Blob<double> x (5, 2);
		x.setName("x");
		x.value.rand_();
		Blob<double> y (5, 2);
		y.setName("y");
		SoftmaxLayer<double> sm1;
		sm1.forward(x, y);
		y.gradient.fill_(1.);
		sm1.backward(x, y);
		x.dump();
		y.dump();
		y.gradient.rand_();
		sm1.backward(x, y);
		x.dump();
		y.dump();
	}; TE;

	TS("classnllloss layer"); {
		SoftmaxLayer<double> sm1;
		Blob<double> yhat (5, 2);
		yhat.setName("yhat");
		yhat.value.rand_();
		sm1.forward(yhat, yhat);
		Blob<double> y (1, 2, false);
		y.setName("y");
		y.value.fill_(1.);
		Blob<double> loss (1);
		ClassNLLLoss<double> loss1;
		loss1.forward(yhat, loss, y);
		loss1.report();
		loss1.backward(yhat, loss, y);
		y.dump();
		yhat.dump();
	}; TE;

	TS("classaccuracy"); {
		ClassAccuracy<double> acc1;
		Blob<double> yhat1 (1, 100);
		Blob<double> yhat2 (1, 100);
		Blob<double> y     (1, 100, false);
		y.value.fill_(1.);
		yhat1.value.fill_(0.);
		yhat2.value.fill_(1.);
		y.setName("y");
		yhat1.setName("yhat1");
		yhat2.setName("yhat2");
		Blob<double> acc (1);

		y.dump(true, false);
		yhat1.dump();
		acc1.forward(yhat1, acc, y);
		acc1.report();

		y.dump(true, false);
		yhat2.dump();
		acc1.forward(yhat2, acc, y);
		acc1.report();
	}; TE;

	return 0;
}
#endif

#include <iostream>
#include <vector>
#include "tensor.cc"

using namespace std;

template <typename Dtype>
class Blob {
public:
	Tensor<Dtype>* value = new Tensor<Dtype>();
	Tensor<Dtype>* gradient = new Tensor<Dtype>();

	Blob(){}

	Blob(int length) {
		this->resize(length);
	}

	Blob(int row, int col) {
		this->resize(row, col);
	}

	void resize(int length) {
		value->resize(length);
		gradient->resize(length);
	}

	void resize(int row, int col) {
		value->resize(row, col);
		gradient->resize(row, col);
	}
};

template <typename Dtype>
class Layer {
public:
	void forward() {}
	void backward() {}
	void update() {}
	void zeroGrad() {}
};

template <typename Dtype>
class LinearLayer : public Layer<Dtype> {
public:
	Blob<Dtype> W;
	Blob<Dtype> b;

	LinearLayer(int dim_dest, int dim_src) {
		W.resize(dim_dest, dim_src);
		b.resize(dim_dest);
		W.value->zero_();
		W.gradient->zero_();
		b.value->zero_();
		b.gradient->zero_();
	}

	void forward(Blob<Dtype> input, Blob<Dtype> output) {
		GEMM(1., W.value, input.value, 0., output.value);
	}
};

#if defined(LITE_TEST_LAYER)
#include "dataloader.cc"
int
main(void)
{
	Blob<double> databatch(12, 10); // d=12, batch10
	Blob<double> out1 (3, 10);
	//databatch.value->fill_(1.);
	databatch.value->rand_();
	cout << "databatch"; databatch.value->dump();

	auto fc1 = LinearLayer<double>(3, 12);
	fc1.W.value->fill_(1.);
	cout << "W"; fc1.W.value->dump();

	fc1.forward(databatch, out1);
	out1.value->dump();
	return 0;
}
#endif

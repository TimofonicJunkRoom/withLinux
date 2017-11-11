#include <iostream>
#include "dataloader.cc"
#include "tensor.cc"
#include "layer.cc"

using namespace std;

#define DUMP(msg, blob, dgrad) if (false) do { \
	cout << msg; \
	blob.dump(true, dgrad); \
} while(0)

int
main(void)
{
	cout << ">> Reading MNIST dataset" << endl;

	Blob<double> trainImages (37800, 784, false);
	lite_hdf5_read("mnist.th.h5", "/train/images", 0, 0, 37800, 784, trainImages.value->data);
	Blob<double> trainLabels (37800, 1, false);
	lite_hdf5_read("mnist.th.h5", "/train/labels", 0, 0, 37800, 1, trainLabels.value->data);

	cout << ">> Initialize Network" << endl;
	Blob<double> image (784, 100, false);
	Blob<double> label (100, false);
	LinearLayer<double> fc1 (384, 784);
	fc1.b.value->fill_(0.01);
	Blob<double> o1 (384, 100);
	ReluLayer<double> relu1;
	LinearLayer<double> fc2 (1, 384);
	fc2.b.value->fill_(0.01);
	Blob<double> yhat (1, 100);
	MSELoss<double> loss1;
	Blob<double> loss (1);

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < 500; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;
		// -- get batch
		image.value = trainImages.value->subTensor_(100*(iteration%377), 100*(iteration%377+1))->transpose();
		image.value->scal_(1./255.);
		label.value = trainLabels.value->subTensor_(100*(iteration%377), 100*(iteration%377+1));
		
		// -- forward
		DUMP("image", image, false);
		fc1.forward(image, o1);
		relu1.forward(o1, o1); // inplace relu
		fc2.forward(o1, yhat);
		loss1.forward(yhat, loss, label);
		// -- report
		loss1.report();
		// -- zerograd
		fc1.zeroGrad();
		o1.zeroGrad();
		fc2.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		fc2.backward(o1, yhat);
		relu1.backward(o1, o1); // inplace relu
		fc1.backward(image, o1);

		fc1.dumpstat();
		//DUMP("W", fc1.W, true);
		//DUMP("b", fc1.b, true);

		//yhat.dump(); // FIXME: Gradient is problematic
		//o.dump();

		// update
		double lr = 1e-3;
		fc1.update(lr);
		fc2.update(lr);
	}

	return 0;
}

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
	Blob<double> o (10, 100);
	Blob<double> yhat (10, 100);
	Blob<double> loss (1);
	Blob<double> accuracy (1);
	LinearLayer<double> fc1 (10, 784);
	SoftmaxLayer<double> sm1;
	ClassNLLLoss<double> loss1;
	ClassAccuracy<double> accuracy1;

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < 500; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;
		// -- get batch
		image.value = trainImages.value->subTensor_(100*(iteration%377), 100*(iteration%377+1))->transpose();
		image.value->scal_(1./255.);
		label.value = trainLabels.value->subTensor_(100*(iteration%377), 100*(iteration%377+1));
		
		// -- forward
		DUMP("image", image, false);
		fc1.forward(image, o);
		DUMP("o", o, true);
		sm1.forward(o, yhat);
		DUMP("yhat", yhat, true);
		loss1.forward(yhat, loss, label);
		DUMP("loss", loss, true);
		accuracy1.forward(yhat, accuracy, label);
		// -- report
		loss1.report();
		accuracy1.report();
		// -- zerograd
		fc1.zeroGrad();
		o.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		sm1.backward(o, yhat);
		fc1.backward(image, o);

		fc1.dumpstat();
		//DUMP("W", fc1.W, true);
		//DUMP("b", fc1.b, true);

		//yhat.dump(); // FIXME: Gradient is problematic
		//o.dump();

		// update
		fc1.update(1e-3);
	}

	return 0;
}

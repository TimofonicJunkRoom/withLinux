#include <iostream>
#include "dataloader.cc"
#include "tensor.cc"
#include "layer.cc"

using namespace std;

int
main(void)
{
	cout << ">> Reading MNIST dataset" << endl;

	Tensor<double> trainImages (37800, 784);
	trainImages.setName("trainImages");
	lite_hdf5_read("mnist.th.h5", "/train/images", 0, 0, 37800, 784, trainImages.data);
	Tensor<double> trainLabels (37800, 1);
	trainLabels.setName("trainLabels");
	lite_hdf5_read("mnist.th.h5", "/train/labels", 0, 0, 37800, 1, trainLabels.data);

	cout << ">> Initialize Network" << endl;

	Blob<double> imageT (784, 100, false);
	imageT.setName("imageT");
	Blob<double> label (1, 100, false);
	label.setName("label");
	Blob<double> o (100, 100);
	o.setName("o");
	Blob<double> yhat (1, 100);
	yhat.setName("yhat");
	Blob<double> loss (1);
	loss.setName("loss");

	LinearLayer<double> fc1 (100, 784);
	ReluLayer<double> relu1;
	LinearLayer<double> fc2 (1, 100);
	MSELoss<double> loss1;

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < 500; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;

		// -- get batch
		Tensor<double>* batchIm = trainImages.sliceRows(100*(iteration%377), 100*(iteration%377+1));
		Tensor<double>* batchLb = trainLabels.sliceRows(100*(iteration%377), 100*(iteration%377+1));
		Tensor<double>* batchImT = batchIm->transpose();
		batchImT->scal_(1./255.);
		imageT.value.copy(batchImT->data, 784*100);
		label.value.copy(batchLb->data, 100);
		
		// -- forward
		fc1.forward(imageT, o);
		relu1.forward(o, o); // inplace relu
		fc2.forward(o, yhat);
		loss1.forward(yhat, loss, label);
		// -- zerograd
		fc1.zeroGrad();
		fc2.zeroGrad();
		o.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		fc2.backward(o, yhat);
		relu1.backward(o, o); // inplace relu
		fc1.backward(imageT, o);
		// -- report
		loss1.report();
		fc1.dumpstat();
		label.dump(true, false);
		yhat.dump(true, false);
		cout << "MAE " << MAE(&label.value, &yhat.value) << endl;
		// update
		double lr = 1e-2;
		fc1.update(lr);
		fc2.update(lr);
	}

	return 0;
}

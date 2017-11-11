/* tensor.cc for LITE
 * Copyright (C) 2017 Mo Zhou <cdluminate@gmail.com>
 * MIT License
 */
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
	Blob<double> o1 (10, 100);
	o1.setName("o1");
	Blob<double> yhat (10, 100);
	yhat.setName("yhat");
	Blob<double> loss (1);
	loss.setName("loss");
	Blob<double> acc (1);
	acc.setName("accuracy");

	LinearLayer<double> fc1 (10, 784);
	SoftmaxLayer<double> sm1;
	ClassNLLLoss<double> loss1;
	ClassAccuracy<double> acc1;


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
		fc1.forward(imageT, o1);
		sm1.forward(o1, yhat);
		acc1.forward(yhat, loss, label);
		loss1.forward(yhat, loss, label);
		// -- zerograd
		fc1.zeroGrad();
		o1.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		sm1.backward(o1, yhat);
		fc1.backward(imageT, o1);
		// -- report
		loss1.report();
		acc1.report();
		fc1.dumpstat();
		//label.dump(true, false);
		//yhat.dump(true, false);
		// update
		double lr = 1e-3;
		fc1.update(lr);

		// cleanup
		delete batchIm;
		delete batchLb;
		delete batchImT;
	}

	return 0;
}

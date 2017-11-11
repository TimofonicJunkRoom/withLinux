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
	Blob<double> o1 (512, 100);
	o1.setName("o1");
	Blob<double> o2 (512, 100);
	o2.setName("o2");
	Blob<double> yhat (1, 100);
	yhat.setName("yhat");
	Blob<double> loss (1);
	loss.setName("loss");

	LinearLayer<double> fc1 (512, 784);
	ReluLayer<double> relu1;
	LinearLayer<double> fc2 (512, 512);
	ReluLayer<double> relu2;
	LinearLayer<double> fc3 (1, 512);
	MSELoss<double> loss1;

	cout << ">> Start training" << endl;
	for (int iteration = 0; iteration < 2000; iteration++) {
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
		relu1.forward(o1, o1); // inplace relu
		fc2.forward(o1, o2);
		relu2.forward(o2, o2); // inplace relu
		fc3.forward(o2, yhat);
		loss1.forward(yhat, loss, label);
		// -- zerograd
		fc1.zeroGrad();
		fc2.zeroGrad();
		fc3.zeroGrad();
		o1.zeroGrad();
		o2.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, label);
		fc3.backward(o2, yhat);
		relu2.backward(o2, o2);
		fc2.backward(o1, o2);
		relu1.backward(o1, o1); // inplace relu
		fc1.backward(imageT, o1);
		// -- report
		loss1.report();
		//fc1.dumpstat();
		label.dump(true, false);
		yhat.dump(true, false);
		cout << "  * MAE " << MAE(&label.value, &yhat.value) << endl;
		// update
		double lr = 1e-1;
		fc1.update(lr);
		fc2.update(lr);
	}

	return 0;
}

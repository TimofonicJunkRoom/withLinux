#include <iostream>
#include "dataloader.cc"
#include "tensor.cc"
#include "layer.cc"

using namespace std;

int
main(void)
{
	cout << "Initialize Test Net" << endl;

	Blob<double> batch (10, 784, false);
	Blob<double> label (10, false);
	Blob<double> o (10, 10);
	Blob<double> yhat (10, 10);
	Blob<double> loss (1);
	Blob<double> accuracy (1);

	LinearLayer<double> fc1 (10, 784);
	SoftmaxLayer<double> sm1;
	ClassNLLLoss<double> loss1;
	ClassAccuracy<double> accuracy1;

	cout << "training" << endl;
	for (int iteration = 0; iteration < 100; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;
		// get batch
		lite_hdf5_read("demo.h5", "data", 0, 0, 10, 784, batch.value->data);
		lite_hdf5_read("demo.h5", "label", 0, 10, label.value->data);
		Blob<double> batchT = *batch.clone();
		batchT.transpose();
		// forward
		fc1.forward(batchT, o);
		sm1.forward(o, yhat);
		loss1.forward(yhat, loss, label);
		accuracy1.forward(yhat, accuracy, label);
		// report
		loss1.report();
		// zerograd
		fc1.zeroGrad();
		// backward
		loss1.backward(yhat, loss, label);
		sm1.backward(o, yhat);
		fc1.backward(batchT, o);

		//yhat.dump(); // FIXME: Gradient is problematic

		// update
		fc1.update(1e+7);
		o.zeroGrad();
		yhat.zeroGrad();
		loss.zeroGrad();

	}

	return 0;
}

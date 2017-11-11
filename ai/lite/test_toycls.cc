#include <iostream>
//#include "dataloader.cc"
#include "tensor.cc"
#include "layer.cc"

using namespace std;

int
main(void)
{
	cout << "Initialize Test Net" << endl;

	Blob<double> batch (50, 200, false);
	batch.setName("batch");
	batch.value.rand_();
	batch.value.add_(-.5);
	Blob<double> label (1, 200, false);
	batch.setName("label");
	for (size_t i = 0; i < label.value.getSize(); i++) {
		*label.value.at(i) = (double)(i%10);
	}
	Blob<double> o (10, 200);
	Blob<double> yhat (10, 200);
	yhat.setName("yhat");
	Blob<double> loss (1);
	Blob<double> accuracy (1);

	LinearLayer<double> fc1 (10, 50);
	SoftmaxLayer<double> sm1;
	ClassNLLLoss<double> loss1;
	ClassAccuracy<double> accuracy1;

	cout << "training" << endl;
	for (int iteration = 0; iteration < 100; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;
		// -- forward
		fc1.forward(batch, o);
		sm1.forward(o, yhat);
		loss1.forward(yhat, loss, label);
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
		fc1.backward(batch, o);

		fc1.dumpstat();
		//DUMP("W", fc1.W, true);
		//DUMP("b", fc1.b, true);

		//yhat.dump(); // FIXME: Gradient is problematic
		//o.dump();

		// update
		fc1.update(1e-1);
	}

	return 0;
}

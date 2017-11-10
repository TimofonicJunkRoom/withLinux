#include <iostream>
//#include "dataloader.cc"
#include "tensor.cc"
#include "layer.cc"

using namespace std;

#define DUMP(msg, blob, dgrad) if (true) do { \
	cout << msg; \
	blob.dump(true, dgrad); \
} while(0)

int
main(void)
{
	cout << "Initialize Test Net" << endl;

	Blob<double> batch (16, 8, false);
	batch.value->rand_();
	Blob<double> label (3, 8, false);
	label.value->rand_();
	Blob<double> o (3, 8);
	Blob<double> loss (1);

	LinearLayer<double> fc1 (3, 16);
	MSELoss<double> loss1;

	DUMP("X", batch, false);
	DUMP("y", label, false);

	cout << "training" << endl;
	//for (int iteration = 0; iteration < 1; iteration++) {
	//for (int iteration = 0; iteration < 5; iteration++) {
	//for (int iteration = 0; iteration < 100; iteration++) {
	for (int iteration = 0; iteration < 2000; iteration++) {
		cout << ">> Iteration " << iteration << "::" << endl;
		// -- forward
		fc1.forward(batch, o);
		//DUMP("o", o, true);
		loss1.forward(o, loss, label);
		//DUMP("loss", loss, true);
		// -- report
		loss1.report();

		Blob<double>* res = o.clone();
		AXPY(-1., label.value, res->value);
		cout << "  * RES ASUM" << res->value->asum() << endl;;

		// -- zerograd
		fc1.zeroGrad();
		o.zeroGrad();
		loss.zeroGrad();
		// -- backward
		loss1.backward(o, loss, label);
		fc1.backward(batch, o);

		//fc1.dumpstat();

		// update
		fc1.update(1e-1);
	}
	DUMP("W", fc1.W, true);
	DUMP("b", fc1.b, true);

	return 0;
}

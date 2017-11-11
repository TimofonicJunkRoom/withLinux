#include <iostream>
//#include "dataloader.cc"
#include "tensor.cc"
#include "layer.cc"

#include "lumin_log.h"

using namespace std;

#define DUMP(msg, blob, dgrad) if (true) do { \
	cout << msg; \
	blob.dump(true, dgrad); \
} while(0)

int
main(void)
{
	// AB = C, given B and C, find A
	double a[4] = {-3.65, -1.175, 1.3, 3.775}; // 1x4
	double b[12] = {1.,5,9,2,6,10,3,7,11,4,8,12}; // 4x3
	double c[3] = {13.,14,15}; // 1x3

	cout << "Initialize Test Net" << endl;

	Blob<double> X (4, 3, false);
	X.value->copy(b, 12);
	Blob<double> y (1, 3, false);
	y.value->copy(c, 3);
	Blob<double> yhat (1, 3);
	Blob<double> loss (1);

	LinearLayer<double> fc1 (1, 4);
	MSELoss<double> loss1;

	DUMP("X", X, false);
	DUMP("y", y, false);
	DUMP("W", fc1.W, false);
	DUMP("b", fc1.b, false);

	for (int iteration = 0; iteration < 5; iteration++) {
		LOG_INFOF (">> Iteration :: %d", iteration);
		// -- forward
		fc1.forward(X, yhat);
		loss1.forward(yhat, loss, y);
		// -- report
		loss1.report();

		// -- zerograd
		yhat.zeroGrad();
		loss.zeroGrad();
		fc1.zeroGrad();
		// -- backward
		loss1.backward(yhat, loss, y);
		fc1.backward(X, yhat);

		// update
		fc1.update(1e-2);
	}
	DUMP("W", fc1.W, true);
	DUMP("b", fc1.b, true);

	return 0;
}

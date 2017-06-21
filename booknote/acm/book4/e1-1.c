#include <stdio.h>

// get average
double
average(int *v, size_t sz) {
	double sum = .0;
	for (int i = 0; i < sz; i++) {
		sum += (double)(v[i]);
	}
	return sum/sz;
}

int
main (void) {
	int vector[] = { 1,2,3 };
	printf("%.3lf\n", average(vector, 3));
	return 0;
}

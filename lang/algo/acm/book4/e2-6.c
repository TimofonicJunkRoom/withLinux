#include <stdio.h>
#include <math.h>

double
harmony(int n)
{
	double sum = 0.;
	for (int i = n; i > 0; i--) 
		sum += 1. / i;
	return sum;
}

int
main(void)
{
	printf("%lf\n", harmony(3));
	return 0;
}

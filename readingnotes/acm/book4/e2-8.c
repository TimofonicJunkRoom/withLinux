#include <stdio.h>
#include <math.h>

double
subsequence(int n, int m)
{
	double sum = 0.;
	for (int i = m; i >= n; i--) {
		sum += 1./((double)i*i);
	}
	return sum;
}

int
main(void)
{
	printf("%lf %lf\n", subsequence(2,4), subsequence(65536, 655360));
	return 0;
}

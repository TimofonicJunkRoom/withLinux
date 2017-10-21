#include <stdio.h>
#include <limits.h>

// data statics, input=stdin
void
datastat(void) {
	int x = 0, count = 0, sum = 0, max = INT_MIN, min = INT_MAX;
	while (scanf("%d", &x) == 1) {
		min = (x < min) ? x : min;
		max = (x > max) ? x : max;
		sum += x;
		count ++;
	}
	printf("%d %d %f\n", min, max, (float)sum/count);
	return;
}

int
main(void)
{
	datastat();
	return 0;
}

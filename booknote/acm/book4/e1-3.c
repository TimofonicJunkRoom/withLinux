#include <stdio.h>

long
sum1ton(int n)
{
	long sum = 0;
	for (int i = 1; i <= n; i++)
		sum += i;
	return sum;
}

int
main(void)
{
	printf("%ld\n", sum1ton(10));
	return 0;
}

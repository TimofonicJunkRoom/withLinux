#include <stdio.h>

long
factorial(long n)
{
	return (n == 0) ? 1 : n * factorial(n-1);
}

int
main(void)
{
	printf("%ld\n", factorial(3));
	return 0;
}

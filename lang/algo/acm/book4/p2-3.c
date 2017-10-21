#include <stdio.h>

// sum of factorial
long
factorial(long n) {
	return (0 == n) ? (1) : (1 == n) ? (1) : n * factorial(n-1);
}

long
sumfact(long n) {
	long sum = 0;
	for (int i = 1; i <= n; i++) {
		sum += factorial(i);
	}
	return sum;
}

int
main(void)
{
	printf("%ld", sumfact(10)%1000000);
	return 0;
}

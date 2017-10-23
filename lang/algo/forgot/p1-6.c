#include <stdio.h>

// reverse a given integer

int
main (void)
{
	long n;
	printf("Input integer: ");
	scanf("%ld", &n);
	while ( n > 0 ) {
		printf("%d", n%10);
		n /= 10;
	}
	return 0;
}

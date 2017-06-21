#include <stdio.h>

// number of digits
int
getNumDigits(int n) {
	int counter = 0;
	while (n > 0) {
		n /= 10;
		counter ++;
	}
	return counter;
}

int
main(void)
{
	printf("%d", getNumDigits(12735));
	return 0;
}

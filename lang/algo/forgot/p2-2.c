#include <stdio.h>

// 3n+1 problem
int
p3np1(int n)
{
	if (0 == n) { // boundary 0
		return 0;
	} else if (1 == n) { // boundary 1
		return 0;
	} else {
		// non-boundary
		if (0 == n%2) { // even
			return 1+p3np1(n/2);
		} else { // 1 == n%2 , odd
			return 1+p3np1(3*n + 1);
		}
	}
}

int
main (void)
{
	printf("-> %d\n", p3np1(3));
	return 0;
}

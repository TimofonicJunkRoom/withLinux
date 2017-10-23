#include <stdio.h>
#include <math.h>

// numbers that in format of AABB which are squared numbers
void
aabb(void)
{
	for (int i = 1; i <= 9; i++) for (int j = 0; j <= 9; j++) {
			int n = 1100 * i + 11 * j; // n [1100, 9999]
			//for (int k = 32; k <= 100; k++) { // 32^2 = 1024, 100^2 = 10000
			//	if (k*k == n) {
			//		printf("-> %d\n", n);
			//	}
			//}
			int x = roundf(sqrt(n));
			if (x*x == n) printf("-> %d\n", n);
	}
	return;
}

int main(void) {aabb();}

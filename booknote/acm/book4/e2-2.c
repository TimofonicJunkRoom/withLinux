#include <stdio.h>

void
daffodil(void)
{
#define DAFF(a,b,c) (( a*a*a+b*b*b+c*c*c ))
#define ISDAFF(a,b,c) (( (a*100+b*10+c)==DAFF(a,b,c) ? 1 : 0 ))
	// atoi($A$B$C) == A^2 + B^2 + C^2
	for (int i = 1; i <= 9; i++) {
		for (int j = 0; j <= 9; j++) {
			for (int k = 0; k <= 9; k++) {
				if (ISDAFF(i, j, k)) {
					printf("%d%d%d %d\n", i, j, k, DAFF(i,j,k));
				} else {
					printf("X %d%d%d %d\n", i, j, k, DAFF(i,j,k));
				}
			}
		}
	}
}

int main(void) {daffodil(); }

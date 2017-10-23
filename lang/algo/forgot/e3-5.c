#include <stdio.h>

int
main(void)
{
	int m[3][3] = { 1, 0, 0, 0, 2, 0, 0, 0, 3 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%d ", m[i][j]);
		}
		printf("\n");
	}
	for (int j = 2; j >= 0; j--) {
		for (int i = 0; i < 3; i++) {
			printf("%d ", m[i][j]);
		}
		printf("\n");
	}
	return 0;
}

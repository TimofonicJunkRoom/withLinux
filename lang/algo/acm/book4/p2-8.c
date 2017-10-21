#include <stdio.h>

int
main(void)
{
	int x;
	scanf("%d", &x);
	if (x%2 == 1) {
		printf("odd\n");
	} else {
		printf("even\n");
	}
	return 0;
}

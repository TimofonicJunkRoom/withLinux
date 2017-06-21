#include <stdio.h>

// hanxin
int
main (void)
{
	//int a = 2, b = 1, c = 6;
	int a = 2, b = 1, c = 3;
	int flag_found = 0;
	for (int n = 10; n <= 100; n++) {
		if ((n%3 == a) && (n%5 == b) && (n%7 == c)) {
			flag_found = 1;
			printf("%d\n", n);
		}
	}
	if (!flag_found) {
		printf("NA\n");
	}
	return 0;
}

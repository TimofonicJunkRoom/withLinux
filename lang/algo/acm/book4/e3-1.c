#include <stdio.h>

// stat scores

int
main(void)
{
	int s[8] = {21, 12, 32, 66, 99, 99, 32, 99 }; // or float
	int maxval = 0, maxcount = 0;
	for (int i = 0; i < 8; i++) {
		if (s[i] > maxval) {
			maxval = s[i];
			maxcount = 1;
		} else if (s[i] == maxval) {
			maxcount++;
		}
	}
	printf("score %d count %d\n", maxval, maxcount);
	return 0;
}

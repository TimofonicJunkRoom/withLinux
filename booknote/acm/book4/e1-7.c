#include <stdio.h>

// discount
int
main (void)
{
	int count = 5;
	if (95*count >= 300) {
		printf("%.2f\n", 95*count*.85);
	} else {
		printf("%.2f\n", 95*count);
	}
	return 0;
}

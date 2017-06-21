#include <stdio.h>

// temperature convertion: F -> C
float
tempconv(float f)
{
	return 5.*(f-32.)/9.;
}

int
main(void)
{
	printf("%.3f\n", tempconv(100.));
	return 0;
}

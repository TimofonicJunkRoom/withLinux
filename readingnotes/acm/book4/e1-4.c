#include <stdio.h>
#include <math.h>

#define PI (( 4.*atan(1.0) ))

float
sinfa(float n) { return sinf(n*PI/180.); }

float
cosfa(float n) { return cosf(n*PI/180.); }

int
main(void)
{
	printf("%f %f\n", sinfa(180), cosfa(180));
	return 0;
}

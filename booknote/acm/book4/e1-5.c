#include <stdio.h>
#include <math.h>

struct Point {
	float x;
	float y;
};
typedef struct Point Point;
Point a = { 0., 0. };
Point b = { 1., 1. };

float
NormL2(Point* a, Point* b)
{
	return sqrtf( (a->x - b->x)*(a->x - b->x)+(a->y - b->y)*(a->y - b->y) );
}

int
main (void)
{
	printf("%f\n", NormL2(&a, &b));
	return 0;
}

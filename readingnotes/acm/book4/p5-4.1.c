#include <stdio.h>

int
sum1tok(int k)
{ // sum_{i=1}^k i = k(k+1)/2
	return k*(k+1)/2;
}

void
cantor(int n)
{
	int x = 1;
	while (sum1tok(x) < n) x++;
	int a = sum1tok(x) - n + 1;
	int b = x - a + 1;
	printf(x%2==0 ? "%2$d/%1$d\n" :"%1$d/%2$d\n", a, b);
}

int
main(void)
{
	cantor(3);
	cantor(14);
	cantor(7);
	cantor(12345);
}

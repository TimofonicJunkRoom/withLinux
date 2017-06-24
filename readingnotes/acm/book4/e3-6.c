#include <stdio.h>

void
dectobasebr(int dec, int base)
{
	int remainder = 0;
	while(dec > 0) {
		remainder = dec % base;
		dec = dec / base;
		printf("%d", remainder); // reverse order
	}
	printf("\n");
}

void
dectobaseb(int dec, int base)
{
	int remainder = 0;
	if (dec > 0) {
		remainder = dec % base;
		dectobaseb(dec / base, base); // recursive to make the order correct
		printf("%d", remainder);
	}
}

int
main(void)
{
	dectobasebr(3, 2);
	dectobaseb(10, 2);printf("\n");
	return 0;
}

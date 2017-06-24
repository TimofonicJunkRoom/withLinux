#include <stdio.h>

void
basebtodec(int bb, int base) // from last digit to the first digit
{
	int p = 1;
	int sum = 0;
	while (bb>0) {
		int remainder = bb % 10;
		sum += remainder * p;
		bb = bb / 10;
		p *= base;
	}
	printf("%d\n", sum);
}

/* from the first digit to the last digit : 1010
 * 0x2 + 1 = 1
 * 1x2 + 0 = 2
 * 2x2 + 1 = 5
 * 5x2 + 0 = 10
 * prev answer * 2 + the current digit
 */

int
main(void)
{
	basebtodec(1010, 2);
	basebtodec(11, 2);
	return 0;
}

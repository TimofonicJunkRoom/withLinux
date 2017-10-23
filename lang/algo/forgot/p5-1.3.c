#include <stdio.h>
#include <string.h>

// periodic string

size_t
getPeriod(char* src, size_t sz)
{
	size_t i = 1;
	for (i = 1; i < sz; i++) if (sz % i == 0) {
		int flag = 1;
		for (size_t j = i; j < sz; j++) {
			if (src[j] != src[j % i]) { flag = 0; break; }
		}
		if (flag) { printf("%ld\n", i); break; }
	}
	return i;
}

int
main(void)
{
	char* example = (char*)"HoHoHo";
	printf("%ld\n", getPeriod(example, strlen(example)));
	return 0;
}

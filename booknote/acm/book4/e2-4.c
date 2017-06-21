#include <stdio.h>

// print inversed triangle
void
ptriangle(int n)
{
	for (int i = n; i >= 1; i--) {
		int lensharp = 2*i-1;
		int lenline = 2*n-1;
		int lenpad = (lenline - lensharp)/2;
		for (int j = 0; j < lenpad; j++)
			printf("%c", ' ');
		for (int j = 0; j < lensharp; j++)
			printf("%c", '#');
		for (int j = 0; j < lenpad; j++)
			printf("%c", ' ');
		printf("\n");
	}
	return;
}

int main(void) {ptriangle(5);}

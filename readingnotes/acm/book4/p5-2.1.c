// report the count of carry when adding two numbers together

#include <stdio.h>

int
main(void)
{
	int a, b;
	while (scanf("%d %d", &a, &b)) {
		if (!a && !b) return 0;
		int c = 0, ans = 0;
		for (int i = 9; i >= 0; i--) {
			c = (a%10 + b%10 + c) > 9 ? 1 : 0;
			ans += c;
			a /= 10; b /= 10;
		}
		printf("%d\n", ans);
	}
	return 0;
}

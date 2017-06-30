#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
compare_char(const void* a, const void* b)
{
	return *((const char*)a) - *((const char*)b);
}

int
compare_str(const void* a, const void* b)
{
	return strcmp((const char*)a, (const char*)b);
}

int
main(void)
{
	int n = 0;
	char word[2000][10], sorted[2000][10];

	while (1) {
		scanf("%s", word[n]);
		if (word[n][0] == '*') break;
		n++;
	}
	qsort(word, n, sizeof(word[0]), compare_str);
	for (int i = 0; i < n; i++) {
		strcpy(sorted[i], word[i]);
		qsort(sorted[i], strlen(sorted[i]),
				sizeof(char), compare_char);
	}

	char s[10];
	while (scanf("%s", s) == 1) {
		qsort(s, strlen(s), sizeof(char), compare_char);
		int found = 0;
		for (int i = 0; i < n; i++) {
			if (strcmp(sorted[i], s) == 0) {
				found = 1;
				printf("%s ", word[i]);
			}
		}
		if (!found) printf(":(");
		printf("\n");
	}
	return 0;
}
/*
 * tarp given score refund only trap work earn course pepper part
 * ******
 * resco
 * score
 * nfudre aptr sett oresuc
 * refund
 * part tarp trap
 * :(
 * course
 */

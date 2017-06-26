#include <stdio.h>
#include <string.h>

// TeX quotation

void
toTex(char* src, size_t sz)
{
	char* pc = src;
	int in_quote = 0;
	while (*pc != '\0') {
		if (*pc == '"') {
			printf("%s", in_quote ? "''" : "``");
			in_quote = !in_quote;
		} else {
			printf("%c", *pc);
		}
		pc++;
	}
	return;
}

int
main(void)
{
	char* example = (char*)"\"to be or not to be\", quoth the Bard, \"that is the question\".";
	toTex(example, strlen(example));
	return 0;
}

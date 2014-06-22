main:
	gcc -Wall -o a8freq a8freq.c
	gcc -Wall -o a8lu a8lu.c
	gcc -Wall -o a8shift a8shift.c
purge:
	rm a8freq a8lu a8shift

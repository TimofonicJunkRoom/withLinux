CC=/usr/bin/gcc -Wall

main:
	${CC} -o a8freq a8freq.c
	${CC} -o a8lu a8lu.c
	${CC} -o a8shift a8shift.c
purge:
	rm a8freq a8lu a8shift

CC=/usr/bin/gcc -Wall

main:
	${CC} -o a8lu a8lu.c
	${CC} -o a8shift a8shift.c
	make bf	
bf:
	gcc -g -Wall -fopenmp -o bytefreq bytefreq.c
purge:
	rm bytefreq a8shift a8lu

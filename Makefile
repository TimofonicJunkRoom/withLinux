CC=/usr/bin/gcc -Wall

bf:
	gcc -g -Wall -fopenmp -o bytefreq bytefreq.c
purge:
	rm bytefreq

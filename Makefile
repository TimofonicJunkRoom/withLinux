CC=/usr/bin/gcc -Wall

main:
	gcc -g -Wall -fopenmp -o bytefreq bytefreq.c
purge:
	rm bytefreq

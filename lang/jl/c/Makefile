CC=gcc
main:
	$(CC) -g -Wall -fopenmp -shared -fPIE kernel.c -o libkernel.so
	$(CC) -g -Wall -shared -fPIE boxmuller.c -o libboxmuller.so
clean:
	-rm libkernel.so libboxmuller.so

CC=gcc
main:
	$(CC) -g -Wall -fopenmp -shared -fPIE kernel.c -o libkernel.so
	$(CC) -g -Wall -shared -fPIE boxmuller.c -o libboxmuller.so
	export LD_LIBRARY_PATH=.; ./main.jl
clean:
	-rm libkernel.so libboxmuller.so

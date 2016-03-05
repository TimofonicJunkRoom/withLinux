CC=gcc
.PHONY: main static shared demo demo_static demo_shared clean all
main: demo

static:
	$(CC) -g -c -Wall libcdalog.c
shared:
	$(CC) -shared -g -fPIC -Wall libcdalog.c -o libcdalog.so
demo: demo_static
demo_static: static
	$(CC) -g -Wall -o demo_static demo.c libcdalog.o
demo_shared: shared
	$(CC) -g -Wall -o demo_shared demo.c -L. -lcdalog
clean:
	-$(RM) *.o *.so demo_static demo_shared
all: demo_static demo_shared

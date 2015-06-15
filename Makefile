CFLAGS := -g -O2 -Wall 

main:
	$(CC) $(CFLAGS) -c cda.c
	$(CC) $(CFLAGS) -o cda cda.o
clean:
	$(RM) *.o cda
install: main
	install -m 0755 cda /usr/bin/

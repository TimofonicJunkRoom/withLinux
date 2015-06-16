CFLAGS := -g -O2 -Wall 

main:
	$(MAKE) cda
%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<
cda: cda.o
	$(CC) $(CFLAGS) $< -o cda 
	
clean:
	$(RM) *.o cda
install: main
	install -m 0755 cda /usr/bin/

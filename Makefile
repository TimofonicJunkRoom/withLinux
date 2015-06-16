CFLAGS  := -g -O2 -Wall 
INSTALL := install
DESTDIR :=
BIN     := /usr/bin/

main: cda
%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<
cda: cda.o
	$(CC) $(CFLAGS) $< -o cda 
	
clean:
	$(RM) *.o cda
install: main
	$(INSTALL) -m 0755 cda $(DESTDIR)/$(BIN)

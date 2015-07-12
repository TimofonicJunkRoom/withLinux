CFLAGS  := -g -O2 -Wall 
INSTALL := install
DESTDIR :=
BIN     := /usr/bin/

main: libchain.so
%.o: %.c %.h
	$(CC) $(CFLAGS) -fPIC -c $<
libchain.so: chain.o
	$(CC) -shared $(CFLAGS) $< \
		-o libchain.so.0 -Wl,-soname,libchain.so.0
	ln -s libchain.so.0 libchain.so
	
clean:
	$(RM) *.o libchain.so*

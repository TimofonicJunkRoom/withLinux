CFLAGS  := -g -O2 -Wall 
INSTALL := install
DESTDIR :=
BIN     := /usr/bin/

OBJ := chain.o 

main: libstack.so

%.o: %.c %.h
	$(CC) $(CFLAGS) -fPIC -c $<

%.h: %.c
	grep -B2 ^{ $< | sed -e 's@{@@g' -e 's/--//g' -e 's/)/);/g' > $@

libstack.so: $(OBJ)
	$(CC) -shared $(CFLAGS) $< \
		-o libstack.so.0 -Wl,-soname,libstack.so.0
	ln -s libstack.so.0 libstack.so
	
clean:
	$(RM) *.o libstack.so*

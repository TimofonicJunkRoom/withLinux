main:
	gcc -shared -I/usr/include/lua5.1 -I. -lTH -lluaT -llua5.1 thnnauxmod.c \
		-o libthnnauxmod.so -Wall -fPIC
clean:
	$(RM) *.so


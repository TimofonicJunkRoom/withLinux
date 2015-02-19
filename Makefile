DESTDIR=
BIN=$(DESTDIR)/usr/bin

CC=/usr/bin/gcc -Wall

main:
	make bf
bf:
	gcc -O1 -Wall -fopenmp -o bytefreq bytefreq.c
	strip bytefreq
clean:
	rm bytefreq
install: bf
	install -m0755 bytefreq $(BIN)/ 
uninstall:
	rm /usr/bin/bytefreq 
deb-pkg:
	dpkg-buildpackage -us -uc

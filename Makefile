DESTDIR=
BIN=$(DESTDIR)/usr/bin

CC=/usr/bin/gcc -Wall

bytefreq: bytefreq.c
	gcc -O1 -Wall -fopenmp -o bytefreq bytefreq.c
install: bytefreq
	install -m0755 bytefreq $(BIN)
uninstall:
	rm /usr/bin/bytefreq 
deb-pkg:
	dpkg-buildpackage -us -uc
	make clean
clean:
	-rm bytefreq
#	if [ -e bytefreq ]; then\
#		rm bytefreq; \
#	fi

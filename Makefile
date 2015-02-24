DESTDIR=
BIN=$(DESTDIR)/usr/bin

CC=/usr/bin/gcc -Wall

main:
	make bf
bf:
	gcc -O1 -Wall -fopenmp -o bytefreq bytefreq.c
clean:
	if [ -e bytefreq ]; then\
		rm bytefreq; \
	fi
install: bf
	install -m0755 bytefreq $(BIN)
uninstall:
	rm /usr/bin/bytefreq 
deb-pkg:
	dpkg-buildpackage -us -uc

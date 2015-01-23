CC=/usr/bin/gcc -Wall

main:
	make bf
bf:
	gcc -O1 -Wall -fopenmp -o bytefreq bytefreq.c
	strip bytefreq
purge:
	rm bytefreq
install:
	strip bytefreq
	cp bytefreq /usr/bin/
	chmod 0755 /usr/bin/bytefreq
uninstall:
	rm /usr/bin/bytefreq 
help:
	@echo "N/A : compile\npurge : delete binary file\ninstall : install bytefreq into /usr/bin/\nuninstall : ..\nhelp : show this message\n"

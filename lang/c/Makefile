db: clean
	gcc -g -Wall db.c -o db -ldb
clean:
	-rm db _temp.db
ec:
	gcc mount.ecryptfs.c -o mount.ecryptfs -O2 -g -Wall -pedantic
	strip mount.ecryptfs
	chown root:root mount.ecryptfs
	chmod u+s mount.ecryptfs

/* bytefreq.c

   cdluminate
 */

#include "crunch.c"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

long counter[256];
int fd;
int loop;

long (* Crunch)(int _fd, long _counter[256]);

int
main (int argc, char **argv)
{
	Crunch = crunch_serial;

	if (argc != 2) {
		printf ("usage : %s FILE\n", argv[0]);
		exit (1);
	}

	if ((fd = open (argv[1], O_RDONLY)) == -1) {
		perror ("open");
		exit (1);
	}

	fputs ("Crunching data ...\n", stderr);
	Crunch (fd, counter);
	fputs ("Dumping data ...\n", stderr);

	for (loop = 0; loop < 256; loop++) {
		if (counter[loop]) printf ("%0x : %ld\n", loop, counter[loop]);
	}
	
	return 0;
}

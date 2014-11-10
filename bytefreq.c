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

long (* Crunch)(int _fd, long _counter[256]);

int
main (int argc, char **argv)
{
	Crunch = crunch_serial;

	if (argc != 2) {
		exit (1);
	}

	if ((fd = open (argv[1], O_RDONLY)) == -1) {
		perror ("open");
		exit (1);
	}

	fputs ("Crunching data ...\n", stderr);
	Crunch (fd, counter);
	
	write (1, counter, 256*sizeof(long));

	return 0;
}

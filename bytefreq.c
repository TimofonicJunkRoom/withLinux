/* bytefreq.c

   Count Byte/Char freqency, using Serial/Parallel Approaches.

   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */

// TODO add getopt
// TODO add commemt

#include "crunch.c"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ================================================= */
void Usage (char *pname)
{
	fprintf (stderr,
"bytefreq : Count Byte freqency in Serial/Parallel approach.\n"
"Author: C.D.Luminate / MIT Licence / 2014\n"
"Usage:\n"
"   %s [options] file\n"
"options:\n"
"  under dev ...\n", pname);
}

/* ================================================= */
long counter[256]; /* counter for bytes */
long total_read;

int fd; /* for open */
int loop;

long (* Crunch)(int _fd, long _counter[256]);

int
main (int argc, char **argv)
{
	Crunch = crunch_serial;

	if (argc != 2) {
		Usage (argv[0]);
		exit (1);
	}

	if ((fd = open (argv[1], O_RDONLY)) == -1) {
		perror ("open");
		exit (1);
	}

	fputs ("Crunching data ...\n", stderr);
	total_read = Crunch (fd, counter);
	fputs ("Dump data ...\n", stderr);

	for (loop = 0; loop < 256; loop++) {
		if (counter[loop]) printf ("%0x : %ld\n", loop, counter[loop]);
	}
	printf ("Read %ld in total\n", total_read);
	
	return 0;
}

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
"Usage:\n"
"  %s [options] file\n"
"  (More info see -v)\n"
"options:\n"
"  -h show this help message\n"
"  -v show version info\n"
"  -p use parallel approach\n"
"  ...\n", pname);
}

void Version (char *pname)
{
	fprintf (stderr,
"Version info of %s :\n"
"Count Byte freqency in Serial/Parallel approach.\n"
"Author: C.D.Luminate / MIT Licence / 2014\n"
"Version: developing\n", pname);
}
/* ================================================= */
long counter[256]; /* counter for bytes */
long total_read;

int fd; /* for open */
int loop;
int opt;

long (* Crunch)(int _fd, long _counter[256]);

int
main (int argc, char **argv)
{
	/* use Serial approach as default */
	Crunch = crunch_serial;
	/* need a test, see TODO in crunch.c */
	//Crunch = crunch_parallel;

	/* parse option */
	while ((opt = getopt(argc, argv, "pvh")) != -1) {
		switch (opt) {
		case 'p':
			/* use parallel */
			Crunch = crunch_parallel;
			break;
		case 'h':
			/* help */
			Usage (argv[0]);
			exit (EXIT_SUCCESS);
			break;
		case 'v':
			/* version info */
			Version (argv[0]);
			exit (EXIT_SUCCESS);
			break;
		default:
			Usage (argv[0]);
			exit (EXIT_FAILURE);
		}
	}
	/* see if user gave a file */
	if (optind >= argc) {
		fprintf (stderr, "%s: a file must be specified.\n", argv[0]);
		Usage (argv[0]);
		exit (EXIT_FAILURE);
	}
	/* open file, then pass the fd to Crunch() */
	if ((fd = open (argv[optind], O_RDONLY)) == -1) {
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

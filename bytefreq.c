/* bytefreq.c

   Count Byte/Char freqency, using Serial/Parallel Approaches.

   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */

// TODO add getopt
// TODO add commemt

#include "crunch.c"
#include "mark.h"

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
"options:\n"
"  -h show this help message\n"
"  -v show version info\n"
"  -p use parallel approach\n"
"  ...\n"
"  for more info see -v\n", pname);
}

void Version (char *pname)
{
	fprintf (stderr,
"Version: developing\n"
"%s : Count Byte freqency in Serial/Parallel approach.\n"
"Author: C.D.Luminate / MIT Licence / 2014\n", pname);
}
/* ================================================= */

long counter[256]; /* counter for bytes, these are raw data */
struct countgrp {
	/* deal with raw data */
	long control;
	long symbol;
	long number;
	long upper;
	long lower;
} cntgrp;
int count_mark[256]; /* 1 if specified by user, or 0 */

long total_read;

int fd; /* for open */
int loop;
int opt; /* for getopt() */

/* used to select a crunch_* function */
long (* Crunch)(int _fd, long _counter[256]);

int
main (int argc, char **argv)
{
	/* Serial as default, use -p to switch to parallel */
	Crunch = crunch_serial;

	bzero (count_mark, sizeof(count_mark));

	// temporary
	_count_marker (256, count_mark);

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
		case 'u':
			/* upper */
		case 'l':
			/* lower */
		case 'A':
			/* all */
		case 'a':
			/* alphabets, i.e. upper && lower */
		case 's':
			/* symbol */
		case 'n':
			/* number */
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
		if (count_mark[loop]) printf ("%0x : %ld\n", loop, counter[loop]);
	}
	printf ("Read %ld in total\n", total_read);
	
	return 0;
}

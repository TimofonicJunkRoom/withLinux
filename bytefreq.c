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
"Description:\n"
"  Count the frequency of specified char.\n"
"  Only shows Total read size if no char specified.\n"
"Options:\n"
"  -h show this help message\n"
"  -V show version info\n"
"  -p use parallel approach\n"
"  -A specify all bytes to count\n"
"  -l specify lower to count\n"
"  -u specify upper to count\n"
"  -s specify symbol to count\n"
"  -c specify control character to count\n"
"  -a specify alphabets to count (= '-lu')\n"
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
int count_mark[256]; /* 1 if specified by user, or 0 */
long total_read;

struct {
	/* deal with raw data */
	long control;
	long symbol;
	long number;
	long upper;
	long lower;
	/* cooked data */
	long visible;
	long invisible;
	long alpha;
} counterwrap; /* raw counter wrapp */

struct {
	long total_spec;
	long total_byte;
} countertot;

struct bytefreq_ex {
	/* store the extreme values */
	long spec_max;
	long spec_min;
	long byte_max;
	long byte_min;
	char spec_max_char;
	char spec_min_char;
	char byte_max_char;
	char byte_min_char;
} extr;

int fd; /* for open */
int loop;
int opt; /* for getopt() */

/* used to select a crunch_* function */
long (* Crunch)(int _fd, long _counter[256]);
int no_mark_set (int _mark[256]);
void find_spec_extreme (struct bytefreq_ex *_ex, int _mark[256], long _counter[256]);
void find_byte_extreme (struct bytefreq_ex *_ex, long _counter[256]);

/* ----------------------------------------------------------- */
int
main (int argc, char **argv)
{
	/* Serial as default, use -p to switch to parallel */
	Crunch = crunch_serial;

	bzero (count_mark, sizeof(count_mark));

	/* parse option */
	while ((opt = getopt(argc, argv, "hVpulAasn")) != -1) {
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
		case 'V':
			/* version info */
			Version (argv[0]);
			exit (EXIT_SUCCESS);
			break;
		case 'u':
			/* upper */
			mark_upper (count_mark);
			break;
		case 'l':
			/* lower */
			mark_lower (count_mark);
			break;
		case 'A':
			/* all */
			mark_all (count_mark);
			break;
		case 'a':
			/* alphabets, i.e. upper && lower */
			mark_lower (count_mark);
			mark_upper (count_mark);
			break;
		case 's':
			/* symbol */
			mark_symbol (count_mark);
			break;
		case 'n':
			/* number */
			mark_number (count_mark);
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
	/* see marks */
	if (no_mark_set (count_mark)) {
		fprintf (stderr,
"HINT: see -h to find out options.\n");
	}

	/* ###### start Crunch ########## */
	fputs ("\x1B[mCrunching data ...\n", stderr);
	total_read = Crunch (fd, counter);

	/* ###### cook the raw counter ##### */
	/* find minmax */
	find_byte_extreme (&extr, counter);
	find_spec_extreme (&extr, count_mark, counter);

	/* TODO optimize printer as a8freq's */
	for (loop = 0; loop < 256; loop++) {
		if (!count_mark[loop])
			continue;

		if (loop == extr.spec_max_char)
			fprintf (stdout, "\x1B[31m");
		if (loop == extr.spec_min_char)
			fprintf (stdout, "\x1B[32m");

		fprintf (stdout, "%0x : %ld\x1B[m\n", loop, counter[loop]);
	}

	/* ###### summary ####### */
	fprintf (stdout, "Maximous of specified : (0x%X  %c) : \x1B[33m%ld\x1B[m\n",
		 extr.spec_max_char, extr.spec_max_char, extr.spec_max);
	fprintf (stdout, "Minimous of specified : (0x%X, %c) : \x1B[33m%ld\x1B[m\n",
		 extr.spec_min_char, extr.spec_min_char, extr.spec_min);
	fprintf (stdout, "Total read() : \x1B[33m%ld\x1B[m\n", total_read);
	
	return 0;
}

int
no_mark_set (int _mark[256])
{
	/* if no mark is set, return 1 (true), or return 0(false) */
	int _lo;
	for (_lo = 0; _lo < 256; _lo++) {
		if (_mark[_lo] > 0) return 0;
	}
	return 1;
}

void
find_byte_extreme (struct bytefreq_ex *_ex, long _counter[256])
{
	int _lo;
	long _max = 0;
	long _min = 0;
	char _maxc = 0;
	char _minc = 0;

	for (_lo = 0; _lo < 256; _lo++) {
		if (_counter[_lo] > _max) {
			_max = _counter[_lo];
			_maxc = (char)_lo;
		}
	}
	_min = _max; /* important ! */
	for (_lo = 0; _lo < 256; _lo++) {
		if (_counter[_lo] < _min) {
			_min = _counter[_lo];
			_minc = (char)_lo;
		}
	}

	_ex -> byte_max = _max;
	_ex -> byte_min = _min;
	_ex -> byte_max_char = _maxc;
	_ex -> byte_min_char = _minc;
	return;
}

void
find_spec_extreme (struct bytefreq_ex *_ex, int _mark[256], long _counter[256])
{
	int _lo;
	long _max = 0;
	long _min;
	char _maxc = 0;
	char _minc = 0;

	for (_lo = 0; _lo < 256; _lo++) {
		if (_counter[_lo] > _max && _mark[_lo])
		{
			_max = _counter[_lo];
			_maxc = (char)_lo;
		}
	}
	_min = _max; /* important ! */
	for (_lo = 0; _lo < 256; _lo++) {
		if (_counter[_lo] < _min && _mark[_lo]) {
			_min = _counter[_lo];
			_minc = (char)_lo;
		}
	}

	_ex -> spec_max = _max;
	_ex -> spec_min = _min;
	_ex -> spec_max_char = _maxc;
	_ex -> spec_min_char = _minc;
	return;
}

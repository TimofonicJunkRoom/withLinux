/* bytefreq.c

   Count Byte/Char freqency, using Serial/Parallel Approaches.

   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */

// TODO : -S option does not accept 0x?? hex number (atoi)
/* FIXME : when user's file size exceeds max_of(long), overflow. */

#define BYTEFREQ_VERSION "Version: 2.2 (2014/11/24)\n"

#include "crunch.h"
#include "mark.h"
#include "struct.h"

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
"  %s [options] [FILE]\n"
"Description:\n"
"  Count frequency of specified set of Bytes/Char.\n"
"  Only shows Total read size if no char specified.\n"
"  If given no <FILE>, it would read from the stdin.\n"
"Options:\n"
"  -h     show this help message\n"
"  -V     show version info\n"
"  -v     verbose mode\n"
"  -d     don't use percent output, use float instead\n"
"\n"
"  -p     use parallel approach\n"
"  -U     use UNIX socket apprach (sendfile)\n"
"  -A     specify all bytes to count\n"
"  -l     specify lower to count\n"
"  -u     specify upper to count\n"
"  -s     specify symbol to count\n"
"  -c     specify control character to count\n"
"  -a     specify alphabets to count (= '-lu')\n"
"  -S <N> specify the byte N (decimal)\n"
"  ...\n"
"For more info see -v\n", pname);
}

void Version (char *pname)
{
	fprintf (stderr,
BYTEFREQ_VERSION
"%s : Count Byte freqency in different approaches.\n"
"Author: C.D.Luminate / MIT Licence / 2014\n", pname);
}
/* ================================================= */

long counter[256]; /* counter for bytes, these are raw data */
int count_mark[256]; /* 1 if specified by user, or 0 */
long total_read;

int fd; /* for open */
int loop;
int opt; /* for getopt() */
int dont_use_percent_output;
int use_stdin;
int verbose;

/* used to select a crunch_* function */
long (* Crunch)(int _fd, long _counter[256], int _verbose);

int no_mark_set (int _mark[256]);
void find_spec_extreme (struct bytefreq_ex *_ex, int _mark[256], long _counter[256]);
void find_byte_extreme (struct bytefreq_ex *_ex, long _counter[256]);
void find_total (struct bytefreq_tot *_tot, int _mark[256], long _counter[256]);
int expection (long _counter[256]);

/* ----------------------------------------------------------- */
int
main (int argc, char **argv)
{
	/* Serial as default, use -p to switch to parallel */
	Crunch = crunch_serial;

	/* clear structure */
	bzero (&bf, sizeof(bf));

	/* parse option */
	while ((opt = getopt(argc, argv, "hVpulAasndvcS:U")) != -1) {
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
		case 'v':
			/* verbose mode */
			verbose = 1;
			break;
		case 'c':
			mark_control (bf.mark);
			break;
		case 'u':
			/* upper */
			mark_upper (bf.mark);
			break;
		case 'l':
			/* lower */
			mark_lower (bf.mark);
			break;
		case 'A':
			/* all */
			mark_all (bf.mark);
			break;
		case 'a':
			/* alphabets, i.e. upper && lower */
			mark_lower (bf.mark);
			mark_upper (bf.mark);
			break;
		case 's':
			/* symbol */
			mark_symbol (bf.mark);
			break;
		case 'n':
			/* number */
			mark_number (bf.mark);
			break;
		case 'd':
			/* don't use percent output */
			dont_use_percent_output = 1;
			break;
		case 'S':
			/* specify a byte (decimal) to count */
			if (atoi(optarg) > 255 || atoi(optarg) < 0) {
				fprintf (stderr, "%s: Specified an invalid byte.\n", argv[0]);
				exit (EXIT_FAILURE);
			}
			bf.mark[(unsigned int)atoi(optarg)] = 1;
			break;
		case 'U':
			/* use crunch_unixsock */
			Crunch = crunch_unixsock;
			break;
		default:
			Usage (argv[0]);
			exit (EXIT_FAILURE);
		}
	}
	/* see if user want to use stdin */
	if (optind >= argc) {
		use_stdin = 1;
		fd = fileno(stdin);
	}
	/* open file, then pass the fd to Crunch() */
	if (!use_stdin) {
		if ((fd = open (argv[optind], O_RDONLY)) == -1) {
			perror ("open");
			exit (EXIT_FAILURE);
		}
	}
	/* see marks */
	if (no_mark_set (bf.mark)) {
		fprintf (stderr,
"HINT: see -h to find out more options.\n");
	}

	/* ###### start Crunch ########## */
	if (verbose) fputs ("\x1B[mCrunching data ...\n", stderr);
	total_read = Crunch (fd, bf.c, verbose);

	/* ###### cook the raw counter ##### */
	/* find minmax */
	find_byte_extreme (&(bf.ex), bf.c);
	find_spec_extreme (&(bf.ex), bf.mark, bf.c);
	find_total (&(bf.tot), bf.mark, bf.c);

	/* print the table header */
	fprintf (stdout,
"===========================================================\n"
"Character      Count           of_ALL          of_Specified\n"
"===========    ============    ============    ============\n");

	/* print info about specified chars */
	for (loop = 0; loop < 256; loop++) {
		if (!(bf.mark[loop]))
			continue;

		if (bf.c[loop] == bf.ex.spec_max)
			fprintf (stdout, "\x1B[31m");
		if (bf.c[loop] == bf.ex.spec_min)
			fprintf (stdout, "\x1B[32m");

		if (dont_use_percent_output)
			fprintf (stdout, "(0x%1$x, '%2$c')    %3$12ld    %5$12.8lf    %4$12.8lf\n", loop, loop,
				 bf.c[loop], (double)bf.c[loop]/bf.tot.total_spec,
				 (double)bf.c[loop]/bf.tot.total_byte);
		else
			fprintf (stdout, "(0x%1$x, '%2$c')    %3$12ld   %5$11.3lf %%   %4$11.3lf %%\n", loop, loop,
				 bf.c[loop], (double)100.0*bf.c[loop]/bf.tot.total_spec,
				 (double)100.0*bf.c[loop]/bf.tot.total_byte);

		fprintf (stdout, "\x1B[m"); /* restore color */
	}

	/* ###### summary ####### */
	fprintf (stdout, "Maximous of specified : (0x%X  '%c') : \x1B[33m%ld\x1B[m\n",
		 bf.ex.spec_max_char, bf.ex.spec_max_char, bf.ex.spec_max);
	fprintf (stdout, "Minimous of specified : (0x%X, '%c') : \x1B[33m%ld\x1B[m\n",
		 bf.ex.spec_min_char, bf.ex.spec_min_char, bf.ex.spec_min);
	fprintf (stdout, "The Math Expection    : (0x%X, '%c', dec \x1B[33m%d\x1B[m)\n",
		 (char)expection(bf.c), (char)expection(bf.c), expection(bf.c));
	fprintf (stdout, "Total bytes specified : \x1B[33m%ld, %.3lf%%\x1B[m\n",
	 	 bf.tot.total_spec,
		 (double)100.0*bf.tot.total_spec/bf.tot.total_byte);
	fprintf (stdout, "Total bytes read()    : \x1B[33m%ld\x1B[m\n",
	         total_read);
	
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

void
find_total (struct bytefreq_tot *_tot, int _mark[256], long _counter[256])
{
	int _lo;
	for (_lo = 0; _lo < 256; _lo++) {
		_tot -> total_byte += _counter[_lo];
		if (_mark[_lo])
			_tot -> total_spec += _counter[_lo];
	}
	return;
}

int
expection (long _counter[256])
{
	/* calculate the mathematical expection char among the whole counter array */
	long _tot = 0;
	long _cxn = 0;
	int _t;
	for (_t = 0; _t < 256; _t++) {
		_tot += _counter[_t];
		_cxn += _counter[_t] * _t;
	}
	return (int)(_cxn/_tot);
}

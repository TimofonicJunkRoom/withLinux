/* bytefreq.c

   Count Byte/Char freqency, using Serial/Parallel Approaches.

   C.D.Luminate <cdluminate AT 163 DOT com> 
   MIT Licence, 2014
 */

// TODO : -S option does not accept 0x?? hex number (atoi)
/* FIXME : when user's file size exceeds max_of(long), overflow. */

#define BYTEFREQ_VERSION "Version: 2.2 (2014/11/24)\n"

#include "include/crunch.h"
#include "include/mark.h"
#include "include/struct.h"
#include "include/find.h"
#include "include/print.h"

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
"  Count Byte/Char frequency.\n"
"  Only shows Total read size if no char specified.\n"
"  If given no <FILE>, it would read from the stdin.\n"
"Options:\n"
"  -h     show this help message\n"
"  -V     show version info\n"
"  -v     verbose mode\n"
"  -D     debug mode (> -v)\n"
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

void Version (void)
{
	fprintf (stderr,
BYTEFREQ_VERSION
"Author: C.D.Luminate / MIT Licence / 2014\n");
}
/* ================================================= */

long total_read; /* apart from struct bytefreq */

int fd; /* for open */
int loop;
int opt; /* for getopt() */

int use_percent_output;
int use_stdin;
int use_verbose;

/* used to select a crunch_* function */
long (* Crunch)(int _fd, long _counter[256], int _verbose);

/* ----------------------------------------------------------- */
int
main (int argc, char **argv)
{
	/* Serial as default, use -p to switch to parallel */
	Crunch = crunch_serial;
	/* percent output as default */
	use_percent_output = 1;

	/* clear structure */
	bzero (&bf, sizeof(bf));

	/* parse option */
	while ((opt = getopt(argc, argv, "hVpulAasndvcS:UD")) != -1) {
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
			Version ();
			exit (EXIT_SUCCESS);
			break;
		case 'v':
			/* verbose mode */
			use_verbose = 1;
			break;
		case 'D':
			/* debug mode */
			use_verbose = 2;
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
			use_percent_output = 0;
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
	if (find_mark_set (bf.mark) == 0) {
		fprintf (stderr,
"HINT: see -h to find out more options.\n");
	}

	/* ###### start Crunch ########## */
	if (use_verbose) fputs ("\x1B[mCrunching data ...\n", stderr);
	total_read = Crunch (fd, bf.c, use_verbose);

	/* ###### cook the raw counter ##### */
	/* find minmax */
	find_byte_extreme (&(bf.ex), bf.c);
	find_spec_extreme (&(bf.ex), bf.mark, bf.c);
	find_total (&(bf.tot), bf.mark, bf.c);

	/* #### print table #### */
	fprintf (stdout, "\x1B[m");
	print_the_table_header ();

	/* print info about specified chars */
	for (loop = 0; loop < 256; loop++) {
		print_entry (bf, loop, use_percent_output);	
	}

	/* ###### summary ####### */
	print_summary (bf, total_read);

	return 0;
}

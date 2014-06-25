/* a8freq :: a8freq.c
 * 	Simply show the alphabets' freqency in file or stream.
 *
 * Author : C.D.Luminate
 * 	cdluminate AT gmail DOT com
 * 	started at 2014/05/18
 *
 * https://github.com/CDLuminate/a8freq
 * LICENCE : MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void Usage (char *prog_name)
{
	fprintf (stderr, "\
Usage : %s [-hs] [FILE]\n\
Show the alphabets' freqency in file.\n\
If FILE is not specified, stdin would be used.\n\
  -h    Print this help message\n\
  -p	set decimal places in output\n\
  -s    Use another output format\n",
  		 prog_name);
}

/* chooser for a format to print freq. */
int (*printer)(int, unsigned long, double, int);
int freq_print_h (int, unsigned long, double, int); /* print for human */
int freq_print_s (int, unsigned long, double, int); /* print for script */

/* counter[0] is the summary from counter[1] to counter[26]
 * counter[1] is counter of 'a'&&'A'  ...   counter[26] 'z'&&'Z'
 * 
 * On arch of amd64, unsigned long is enough for normal use.
 */
unsigned long counter[27];

int
main (int argc, char **argv)
{
	/* choose printer for human, default */
	printer = freq_print_h;

	int opt = 0;
	int places = 8;
	register char buf = 0;

	/* if user doesn't specify the input FILE,
	 * 	use stdin as default.
	 */
	FILE *in_file = stdin;
	//FILE *out_file = stdout;
	
	
	/* read the options, if user specifies a FILE, read it,
	 * 	or read from stdin
	 */
	while ((opt = getopt(argc, argv, "hp:s")) != -1) {
		switch (opt) {
			case 'h':
				/* help */
				Usage (argv[0]);
				exit (EXIT_SUCCESS);
				break;
			case 'p':
				/* decimal places to print */
				places = atoi (optarg);
				break;
			case 's':
				/* printer for script */
				printer = freq_print_s;
			default:
				/* out of exception */
				Usage (argv[0]);
				exit (EXIT_FAILURE);
				break;
		}
	}


	/* to tell if a FILE is given */
	if (optind < argc) {
		if ((in_file = fopen (argv[optind], "r")) == NULL) {
			perror ("fopen");
			exit(EXIT_FAILURE);
		}
	}


	/* handle stream, core part */
	while ( (buf = fgetc (in_file)) != EOF && !feof(in_file)) {
		/* case 'a' ... 'z', this is a feature supported by gcc.
		 * 	look up gcc doc.
		 * Consider rewrite this block using if...else...
		 * 	if your compiler is not gcc.
		 */
		switch (buf) {
			case 'a' ... 'z':
				counter[0]++;
				counter[1+(buf-'a')]++;
				break;
			case 'A' ... 'Z':
				counter[0]++;
				counter[1+(buf-'A')]++;
				break;
			default:
				break;
		}
	}

	/* show the result */
	int j;
	for (j=1; j < 27; j++) {
		printer ('A'-1+j, counter[j],
			 (double)counter[j]/counter[0], places);
	}

	printf ("ALL %ld alphabets.\n", counter[0]);
	
	fclose (in_file);
	return 0;
}

/* human format print */
int
freq_print_h (/* alphabet */
	      int c,
	      /* its counter */
	      unsigned long cc,
	      /* freq */
	      double freq,
	      /* decimal places */
	      int place)
{
	fprintf (stdout,
		 "%c\t %ld\t %.*lf%% \n",
		 c, cc, place, freq*100.0);
	return 0;
}

/* conviniet format for sorting and script. */
int
freq_print_s (/* alphabet */
	      int c,
	      /* its counter */
	      unsigned long cc,
	      /* freq */
	      double freq,
	      /* decimal places */
	      int place)
{
	fprintf (stdout,
		 "%.*lf\t %ld\t %c \n",
		 place, freq, cc, c);
	return 0;
}

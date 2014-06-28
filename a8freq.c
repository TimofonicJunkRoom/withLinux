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
Char of max_freq will be highlighted in red.\n\
     of min_freq will be highlighted in green.\n\
If FILE is not specified, stdin would be used.\n\
Options :\n\
  -h    Print this help message\n\
  -p	set decimal places in output\n\
  -s    Use another output format\n",
  		 prog_name);
}

/* chooser for a format to print freq. */
int (*printer)(FILE *,int, unsigned long, double, int);
int freq_print_h (FILE *,int, unsigned long, double, int); /* print for human */
int freq_print_s (FILE *,int, unsigned long, double, int); /* print for script */

/* counter[0] is the summary from counter[1] to counter[26]
 * counter[1] is counter of 'a'&&'A'  ...   counter[26] 'z'&&'Z'
 * 
 * On arch of amd64, unsigned long is enough for normal use.
 */
unsigned long counter[27];
/* for finding witch line to highlight in the part of output */
unsigned long counter_max;
unsigned long counter_min;

int
main (int argc, char **argv)
{
	/* choose printer for human, default */
	printer = freq_print_h;

	/* used by getopt () */
	int opt = 0;

	/* decimal places, set it using -p */
	int places = 8;

	/* buffer, with register anyway */
	register char buf = 0;

	/* if user doesn't specify the input FILE,
	 * 	use stdin as default.
	 */
	FILE *in_file = stdin;
	FILE *out_file = stdout;
	
	
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
				break;
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

	/* find out the counter_max */
	int j; /* j used just in for */
	/* give counter_min a value to ensure it is not 0 at the end */
	counter_min = counter[1];
	/* scan counter[] for max and min, counter[0] is ALL so passed */
	for (j=1; j < 27; j++) {
		if (counter[j] > counter_max) {
			counter_max = counter[j];
		}
		if (counter[j] < counter_min) {
			counter_min = counter[j];
		}
	}

	/* next, print it out, and highlight the max */
	for (j=1; j < 27; j++) {
		/* dye */
		if (counter[j] == counter_max) {
			/* MAX, red */
			fputs ("\033[31m", out_file);
		} else if (counter[j] == counter_min) {
			/* MIN, green */
			fputs ("\033[32m", out_file);
		} else {
			/* no dye */
		}

		/* print info */
		printer (out_file, 'A'-1+j, counter[j],
				 (double)counter[j]/counter[0], places
				);

		/* color recover */
		fputs ("\033[m", out_file);
	}

	/* dump the counter for ALL */
	printf ("ALL \033[34m%ld\033[m alphabets.\n", counter[0]);
	
	fclose (in_file);
	fclose (out_file);
	return 0;
}

/* human format print */
int
freq_print_h (FILE *out_file,
	      /* alphabet */
	      int c,
	      /* its counter */
	      unsigned long cc,
	      /* freq */
	      double freq,
	      /* decimal places */
	      int place)
{
	fprintf (out_file,
		 "%c\t %ld\t %.*lf%% \n",
		 c, cc, place, freq*100.0);
	return 0;
}

/* conviniet format for sorting and script. */
int
freq_print_s (FILE *out_file,
	      /* alphabet */
	      int c,
	      /* its counter */
	      unsigned long cc,
	      /* freq */
	      double freq,
	      /* decimal places */
	      int place)
{
	fprintf (out_file,
		 "%.*lf\t %ld\t %c \n",
		 place, freq, cc, c);
	return 0;
}

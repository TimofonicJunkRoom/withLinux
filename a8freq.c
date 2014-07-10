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

/* FIXME : global, add comment */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


void Usage (char *prog_name)
{
	fprintf (stderr, "\
Usage : %s [-hp:slu] [FILE]\n\
Show the alphabets' freqency in file.\n\
Char of max_freq will be highlighted in red.\n\
     of min_freq will be highlighted in green.\n\
If FILE is not specified, stdin would be used.\n\
The action of \"a8freq -lu\" is different from\n\
	\"a8freq\"\n\
Options :\n\
  -h    Print this help message\n\
  -p	set decimal places in output\n\
  -s    Use another output format\n\
  -l    Custom count the lower\n\
  -u    Custom count the upper\n",
  		 prog_name);
}

/* chooser for a format to print freq. */
int (*printer)(FILE *,int, unsigned long, double, int);
/* print for human */
int freq_print_h (FILE *,int, unsigned long, double, int); 
/* print for script */
int freq_print_s (FILE *,int, unsigned long, double, int); 

/* if_count[N] : if to count this ascii char, (0|1) */
int if_count[256];
int if_count_flush (int this_if_count[256]) {
	int this_looper;
	for (this_looper = 0; this_looper < 256; this_looper++) {
		this_if_count[this_looper] = 0;
	}
	return 0;
}

/* flags for getopt */
int flag_lower;
int flag_upper;
int flag_alpha;
int flag_number;
/*
int flag_punctuation;
int flag_special;
*/

/* counter[N] where N is ascii
 * On arch of amd64, unsigned long is enough for normal use. */
unsigned long counter[256];

/* for finding witch line to highlight in the part of output */
unsigned long counter_max;
unsigned long counter_alpha_max;
unsigned long counter_min;
unsigned long counter_alpha_min;

/* total, the first to leak, custom total */
unsigned long counter_total;
/* counter for whole stream */
unsigned long counter_stream;



int
main (int argc, char **argv)
{
	int looper; /* used in for */

	/* flag_alpha = 1 is default */
	flag_alpha = 1;

	/* init if_count[N] for flag_alpha */
	for (looper = 0; looper < 256; looper++) {
		switch (looper) {
			case 'A' ... 'Z':
			case 'a' ... 'z':
				if_count[looper] = 1;
				break;
			default:
				break;
		}
		//printf ("%c %d ", looper, if_count[looper]);
	}

	/* choose printer for human, default */
	printer = freq_print_h;

	/* used by getopt () */
	int opt = 0;

	/* decimal places, set it using -p */
	int places = 8;

	/* buffer, with register anyway */
	register char buf = 0;

	/* if user doesn't specify the input FILE,
	 * 	use stdin as default.  */
	FILE *in_file = stdin;
	FILE *out_file = stdout;
	
	/* read the options, if user specifies a FILE, read it,
	 * 	or read from stdin */
	while ((opt = getopt(argc, argv, "hlp:su")) != -1) {
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
			case 'l':
				/* set flag_lower */
				flag_alpha = 0;
				
				flag_lower = 1;
				for (looper = 'a'; looper <= 'z'; looper++) {
					if_count[looper] = flag_lower;
				}
				break;
			case 'u':
				/* set flag_upper */
				flag_alpha = 0;

				flag_upper = 1;
				for (looper = 'A'; looper <= 'Z'; looper++) {
					if_count[looper] = flag_upper;
				}
				break;
			default:
				/* out of exception */
				Usage (argv[0]);
				exit (EXIT_FAILURE);
				break;
		}
	}
	/* if flag_alpha was removed, set if_count */
	if (!flag_alpha) {
		if_count_flush (if_count);
		for (looper = 0; looper < 256; looper++) {
			switch (looper) {
				case 'A' ... 'Z':
					if (flag_upper) if_count[looper] = 1;
					break;
				case 'a' ... 'z':
					if (flag_lower) if_count[looper] = 1;
					break;
				default:
					break;
			}
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
	/* FIXME : segfault when input is wide char */
	while ( (buf = fgetc (in_file)) != EOF && !feof(in_file)) {
		counter_stream++;
		counter[(int)buf]++;
		if (if_count[(int)buf]) counter_total++;
	}

	/* find out the counter_max */
	if (!flag_alpha) {
		for (looper = 0; looper < 256; looper++) {
			if (if_count[looper]
			    && counter[looper] > counter_max
			   ) {
				counter_max = counter[looper];
			}
		}
	} else if (flag_alpha) {
		for (looper = 'A'; looper <= 'Z'; looper++) {
			if (if_count[looper]
			    && counter[looper]+counter[looper+32] > counter_max
			   ) {
				counter_max = counter[looper] + counter[looper+32];
			}
		}
	}
	/* then findout the counter_min */
	counter_min = counter_max;
	if (!flag_alpha) {
		for (looper = 0; looper < 256; looper++) {
			if (if_count[looper]
			    && counter[looper] < counter_min
			   ) {
				counter_min = counter[looper];
			}
		}
	} else if (flag_alpha) {
		for (looper = 'A'; looper <= 'Z'; looper++) {
			if (if_count[looper]
			    && counter[looper]+counter[looper+32] < counter_min
			   ) {
				counter_min = counter[looper] + counter[looper+32];
			}
		}
	}
	//printf ("%ld %ld", counter_min, counter_max);

	/* next, print it out, and highlight the max */
	if (!flag_alpha) {
		/* custom mode */
		for (looper = 0; looper < 256; looper++) {
			if (!if_count[looper]) continue;

			/* colour the min/max value */
			if (counter[looper] == counter_max) {
				/* MAX, red */
				fputs ("\033[31m", out_file);
			} else if (counter[looper] == counter_min) {
				/* MIN, green */
				fputs ("\033[32m", out_file);
			}
			printer (out_file, looper, counter[looper],
				 (double)counter[looper]/counter_total, places
				);
			/* color recover */
			fputs ("\033[m", out_file);
		}
	} else if (flag_alpha) {
		/* default mode, flag_alpha */
		for (looper = 0; looper < 256; looper++) {
			if (looper >= 'A' && looper <= 'Z') {
				/* colour the min/max value */
				if (counter[looper]+counter[looper+32] == counter_max) {
					/* MAX, red */
					fputs ("\033[31m", out_file);
				} else if (counter[looper]+counter[looper+32] == counter_min) {
					/* MIN, green */
					fputs ("\033[32m", out_file);
				}

				printer (out_file,
					 looper,
					 counter[looper] + counter[looper+32],
					 (double)(counter[looper]+counter[looper+32])/counter_total,
					 places
					);
				/* color recover */
				fputs ("\033[m", out_file);
			}
		}
	}

	/* dump the counter for ALL */
	fprintf (out_file, "ALL \033[34m%ld\033[m specified chars.\n", counter_total);
	fprintf (out_file, "ALL \033[34m%ld\033[m chars in stream.\n", counter_stream);
	
	/* close file */
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
